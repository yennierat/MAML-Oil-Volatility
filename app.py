"""
Oil Volatility Forecaster — Streamlit Dashboard
─────────────────────────────────────────────────
Live predictions using a MAML-adapted MLP trained on
GDELT geopolitical events + market data.

Requires in the same directory:
  - maml_trained.pth   (MAML meta-learned weights)
  - mlp_pretrained.pth (baseline MLP weights)
  - feature_scaler.pkl (StandardScaler fitted on training data)
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from datetime import datetime, timedelta
import joblib
import requests
import yfinance as yf

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Oil Volatility Forecaster",
    page_icon="🛢️",
    layout="wide"
)

# ── MODEL DEFINITION ────────────────────────────────────
class OilVolatilityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

FEATURE_COLS = [
    'goldstein_scale', 'num_mentions', 'num_sources', 'avg_tone',
    'oil_close', 'oil_vol_5d', 'oil_vol_20d',
    'vix_close', 'ovx_close', 'dxy_close', 'gold_oil_ratio'
]
INNER_LR = 0.01
INNER_STEPS = 5


# ── LOAD MODELS (cached so they only load once) ─────────
@st.cache_resource
def load_models():
    scaler = joblib.load('feature_scaler.pkl')

    maml_model = OilVolatilityMLP()
    maml_model.load_state_dict(torch.load('maml_trained.pth', map_location='cpu'))
    maml_model.eval()

    plain_mlp = OilVolatilityMLP()
    plain_mlp.load_state_dict(torch.load('mlp_pretrained.pth', map_location='cpu'))
    plain_mlp.eval()

    return scaler, maml_model, plain_mlp

scaler, maml_model, plain_mlp = load_models()


# ── MAML ADAPTATION ─────────────────────────────────────
def adapt(model, sup_X, sup_y, inner_lr=INNER_LR, inner_steps=INNER_STEPS):
    adapted = deepcopy(model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(adapted.parameters(), lr=inner_lr)
    adapted.train()
    for _ in range(inner_steps):
        optimizer.zero_grad()
        loss = loss_fn(adapted(sup_X), sup_y)
        loss.backward()
        optimizer.step()
    adapted.eval()
    return adapted


# ── FETCH MARKET DATA ───────────────────────────────────
@st.cache_data(ttl=300)  # refresh every 5 minutes
def fetch_market_data(days_back=30):
    """Fetch oil, VIX, OVX, DXY, gold from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=days_back)

    tickers = {
        'oil_close':  'CL=F',
        'vix_close':  '^VIX',
        'ovx_close':  '^OVX',
        'dxy_close':  'DX-Y.NYB',
        'gold_close': 'GC=F',
    }

    data = {}
    for col, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if len(df) > 0:
                data[col] = df['Close'].squeeze()
        except Exception:
            pass

    if not data:
        return pd.DataFrame()

    market = pd.DataFrame(data)
    market.index = pd.to_datetime(market.index)

    # Compute derived features
    if 'oil_close' in market.columns:
        log_ret = np.log(market['oil_close'] / market['oil_close'].shift(1))
        market['oil_vol_5d'] = log_ret.rolling(5).std() * np.sqrt(252)
        market['oil_vol_20d'] = log_ret.rolling(20).std() * np.sqrt(252)

    if 'gold_close' in market.columns and 'oil_close' in market.columns:
        market['gold_oil_ratio'] = market['gold_close'] / market['oil_close']

    market = market.drop(columns=['gold_close'], errors='ignore')
    return market.dropna()


# ── FETCH GDELT DATA ────────────────────────────────────
@st.cache_data(ttl=900)  # refresh every 15 minutes
def fetch_gdelt_events(hours_back=24):
    """
    Fetch recent GDELT events from the GDELT 2.0 Events API.
    Returns aggregated goldstein_scale, num_mentions, num_sources, avg_tone.
    """
    end = datetime.utcnow()
    start = end - timedelta(hours=hours_back)

    # GDELT 2.0 Events API
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        'query': 'oil OR petroleum OR crude OR OPEC',
        'mode': 'artlist',
        'maxrecords': 250,
        'format': 'json',
        'startdatetime': start.strftime('%Y%m%d%H%M%S'),
        'enddatetime': end.strftime('%Y%m%d%H%M%S'),
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get('articles', [])

        if not articles:
            return None

        tones = [a.get('tone', 0) for a in articles if a.get('tone') is not None]

        return {
            'goldstein_scale': 0.0,  # not available in doc API, use neutral
            'num_mentions': len(articles),
            'num_sources': len(set(a.get('domain', '') for a in articles)),
            'avg_tone': np.mean(tones) if tones else 0.0,
        }

    except Exception:
        # Fallback: use neutral defaults
        return {
            'goldstein_scale': 0.0,
            'num_mentions': 50,
            'num_sources': 10,
            'avg_tone': -1.0,
        }


# ── BUILD FEATURE VECTOR ────────────────────────────────
def build_feature_vector(market_row, gdelt):
    """Combine market + GDELT into the 11-feature vector."""
    features = {
        'goldstein_scale': gdelt['goldstein_scale'],
        'num_mentions':    gdelt['num_mentions'],
        'num_sources':     gdelt['num_sources'],
        'avg_tone':        gdelt['avg_tone'],
        'oil_close':       market_row.get('oil_close', np.nan),
        'oil_vol_5d':      market_row.get('oil_vol_5d', np.nan),
        'oil_vol_20d':     market_row.get('oil_vol_20d', np.nan),
        'vix_close':       market_row.get('vix_close', np.nan),
        'ovx_close':       market_row.get('ovx_close', np.nan),
        'dxy_close':       market_row.get('dxy_close', np.nan),
        'gold_oil_ratio':  market_row.get('gold_oil_ratio', np.nan),
    }
    return np.array([[features[c] for c in FEATURE_COLS]], dtype=np.float32)


# ══════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════

st.title("Oil Volatility Forecaster")
st.caption("MAML-adapted MLP | GDELT geopolitical events + market data")

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    inner_steps = st.slider("MAML adaptation steps", 1, 20, INNER_STEPS)
    inner_lr = st.number_input("Inner learning rate", 0.001, 0.1, INNER_LR, 0.001)
    gdelt_hours = st.slider("GDELT lookback (hours)", 1, 72, 24)
    st.markdown("---")
    st.markdown("**Model files loaded:**")
    st.markdown("- `maml_trained.pth`")
    st.markdown("- `mlp_pretrained.pth`")
    st.markdown("- `feature_scaler.pkl`")

# ── Fetch data ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Market Data")
    with st.spinner("Fetching from Yahoo Finance..."):
        market = fetch_market_data(days_back=60)

    if market.empty:
        st.error("Could not fetch market data. Check your internet connection.")
        st.stop()

    st.dataframe(market.tail(5).round(4), use_container_width=True)

    latest_market = market.iloc[-1].to_dict()
    st.metric("Oil (CL=F)", f"${latest_market.get('oil_close', 0):.2f}")
    st.metric("VIX", f"{latest_market.get('vix_close', 0):.2f}")

with col2:
    st.subheader("GDELT Events")
    with st.spinner("Fetching from GDELT..."):
        gdelt = fetch_gdelt_events(hours_back=gdelt_hours)

    if gdelt is None:
        st.warning("No GDELT data available. Using neutral defaults.")
        gdelt = {
            'goldstein_scale': 0.0,
            'num_mentions': 50,
            'num_sources': 10,
            'avg_tone': -1.0,
        }

    st.metric("Articles found", gdelt['num_mentions'])
    st.metric("Unique sources", gdelt['num_sources'])
    st.metric("Average tone", f"{gdelt['avg_tone']:.2f}")
    st.metric("Goldstein scale", f"{gdelt['goldstein_scale']:.1f}")

# ── Today's prediction ──────────────────────────────────
st.markdown("---")
st.subheader("Today's Volatility Forecast")

raw_features = build_feature_vector(latest_market, gdelt)

if np.isnan(raw_features).any():
    st.error("Some features are missing (NaN). Cannot make prediction.")
    missing = [FEATURE_COLS[i] for i in range(11) if np.isnan(raw_features[0][i])]
    st.write(f"Missing: {missing}")
    st.stop()

scaled_features = scaler.transform(raw_features)
X = torch.tensor(scaled_features, dtype=torch.float32)

# Plain MLP prediction
with torch.no_grad():
    mlp_pred = plain_mlp(X).item()

# MAML prediction (adapt on recent market data as support set)
# Use last 5 days of market data as the "support set" for adaptation
support_data = market.tail(5)
if len(support_data) >= 3:
    sup_features = []
    for _, row in support_data.iterrows():
        f = build_feature_vector(row.to_dict(), gdelt)
        if not np.isnan(f).any():
            sup_features.append(f[0])

    if len(sup_features) >= 2:
        sup_X_raw = np.array(sup_features, dtype=np.float32)
        sup_X_scaled = scaler.transform(sup_X_raw)
        sup_X = torch.tensor(sup_X_scaled, dtype=torch.float32)
        # Use the most recent actual volatility as pseudo-labels
        oil_returns = np.log(support_data['oil_close'] / support_data['oil_close'].shift(1)).dropna()
        realized_vol = oil_returns.rolling(min(len(oil_returns), 3)).std() * np.sqrt(252)
        realized_vol = realized_vol.dropna().values[-len(sup_features):]
        if len(realized_vol) < len(sup_features):
            realized_vol = np.pad(realized_vol,
                                  (len(sup_features) - len(realized_vol), 0),
                                  mode='edge')
        sup_y = torch.tensor(realized_vol, dtype=torch.float32).unsqueeze(1)

        adapted_model = adapt(maml_model, sup_X, sup_y,
                              inner_lr=inner_lr, inner_steps=inner_steps)
        with torch.no_grad():
            maml_pred = adapted_model(X).item()
    else:
        with torch.no_grad():
            maml_pred = maml_model(X).item()
else:
    with torch.no_grad():
        maml_pred = maml_model(X).item()

# Display predictions
pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    st.metric(
        "Plain MLP Forecast",
        f"{mlp_pred:.4f}",
        help="Pretrained MLP without MAML adaptation"
    )

with pred_col2:
    st.metric(
        "MAML Forecast",
        f"{maml_pred:.4f}",
        delta=f"{maml_pred - mlp_pred:+.4f} vs MLP",
        help="MAML-adapted model using recent market data as support set"
    )

with pred_col3:
    vol_level = "LOW" if maml_pred < 0.15 else "MODERATE" if maml_pred < 0.30 else "HIGH"
    st.metric("Volatility Level", vol_level)

# ── Historical chart ─────────────────────────────────────
st.markdown("---")
st.subheader("Historical Market Data")

tab1, tab2, tab3 = st.tabs(["Oil Price", "Volatility", "VIX / OVX"])

with tab1:
    if 'oil_close' in market.columns:
        st.line_chart(market['oil_close'], use_container_width=True)

with tab2:
    vol_cols = [c for c in ['oil_vol_5d', 'oil_vol_20d'] if c in market.columns]
    if vol_cols:
        st.line_chart(market[vol_cols], use_container_width=True)

with tab3:
    indicator_cols = [c for c in ['vix_close', 'ovx_close'] if c in market.columns]
    if indicator_cols:
        st.line_chart(market[indicator_cols], use_container_width=True)

# ── Feature vector debug panel ───────────────────────────
with st.expander("Feature Vector (debug)"):
    feature_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Raw Value': raw_features[0],
        'Scaled Value': scaled_features[0],
    })
    st.dataframe(feature_df, use_container_width=True)

# ── Footer ───────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Data: Yahoo Finance + GDELT 2.0"
)
