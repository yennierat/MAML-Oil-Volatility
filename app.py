"""
Oil Volatility Forecaster — Streamlit Dashboard
─────────────────────────────────────────────────
Live 4h oil volatility predictions using MAML-adapted MLP.
Pulls GDELT data every 15 minutes, makes a prediction,
then after 4 hours computes actual realized vol and plots
actual vs predicted.

Requires in the same directory:
  - maml_trained.pth
  - mlp_pretrained.pth
  - feature_scaler.pkl
  - predictions_log.csv  (auto-created on first run)

Run:
  streamlit run dashboard.py
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
import os

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Oil Vol Forecaster",
    page_icon="O",
    layout="wide"
)

PREDICTIONS_LOG = "predictions_log.csv"

# ── MODEL DEFINITION ─────────────────────────────────────
# Must match Option A training exactly: 15 → 48 → 32 → 16 → 1
class OilVolatilityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 48), nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(48, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# Option A feature set — 15 features (aggregated GDELT + market)
FEATURE_COLS = [
    'ovx_close', 'vix_close', 'oil_vol_5d', 'oil_vol_20d',
    'oil_close', 'dxy_close', 'gold_oil_ratio',
    'gs_mean', 'gs_std', 'gs_conflict_pct', 'gs_weighted',
    'tone_mean', 'tone_std', 'n_events', 'mentions_sum',
]
INNER_LR    = 0.01
INNER_STEPS = 5


# ── LOAD MODELS ──────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler = joblib.load('feature_scaler.pkl')

    maml_model = OilVolatilityMLP()
    maml_model.load_state_dict(
        torch.load('maml_trained.pth', map_location='cpu')
    )
    maml_model.eval()

    plain_mlp = OilVolatilityMLP()
    plain_mlp.load_state_dict(
        torch.load('mlp_pretrained.pth', map_location='cpu')
    )
    plain_mlp.eval()

    return scaler, maml_model, plain_mlp

scaler, maml_model, plain_mlp = load_models()


# ── MAML ADAPTATION ──────────────────────────────────────
def adapt(model, sup_X, sup_y):
    adapted   = deepcopy(model)
    loss_fn   = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.SGD(adapted.parameters(), lr=INNER_LR)
    adapted.eval()
    for _ in range(INNER_STEPS):
        optimizer.zero_grad()
        loss = loss_fn(adapted(sup_X), sup_y)
        loss.backward()
        optimizer.step()
    adapted.eval()
    return adapted


# ── FETCH MARKET DATA ────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_market_data(days_back=30):
    end   = datetime.now()
    start = end - timedelta(days=days_back)

    tickers = {
        'oil_close'  : 'BZ=F',
        'vix_close'  : '^VIX',
        'ovx_close'  : '^OVX',
        'dxy_close'  : 'DX-Y.NYB',
        'gold_close' : 'GC=F',
    }

    data = {}
    for col, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True, timeout=10)
            if len(df) > 0:
                s = df['Close'].squeeze()
                s = s[s > 0]
                if len(s) > 0:
                    data[col] = s
        except Exception:
            pass

    if not data:
        return pd.DataFrame()

    market = pd.DataFrame(data).ffill().bfill()
    market.index = pd.to_datetime(market.index)

    if 'oil_close' in market.columns:
        log_ret = np.log(market['oil_close'] / market['oil_close'].shift(1))
        market['oil_vol_5d']  = log_ret.rolling(5).std()  * np.sqrt(252)
        market['oil_vol_20d'] = log_ret.rolling(20).std() * np.sqrt(252)

    if 'gold_close' in market.columns and 'oil_close' in market.columns:
        market['gold_oil_ratio'] = market['gold_close'] / market['oil_close']

    market = market.drop(columns=['gold_close'], errors='ignore')
    return market.dropna()


# ── FETCH GDELT EVENTS ───────────────────────────────────
# CHANGE 1: use timespan parameter instead of startdatetime/enddatetime
# timespan is far more reliable — the datetime range filter frequently fails
@st.cache_data(ttl=900)
def fetch_gdelt_events(hours_back=24):
    url    = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        'query'      : 'oil OR petroleum OR crude OR OPEC',
        'mode'       : 'artlist',
        'maxrecords' : 250,
        'format'     : 'json',
        'timespan'   : f"{hours_back}h",
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data     = resp.json()
        articles = data.get('articles', [])

        if not articles:
            raise ValueError("Empty response")

        tones    = [float(a['tone']) for a in articles
                    if a.get('tone') is not None]
        n        = len(articles)
        domains  = set(a.get('domain', '') for a in articles if a.get('domain'))
        tone_arr = np.array(tones) if tones else np.array([-1.0])
        gs_approx = tone_arr * 0.5

        return {
            'gs_mean'        : float(np.mean(gs_approx)),
            'gs_std'         : float(np.std(gs_approx, ddof=0)),
            'gs_conflict_pct': float(np.mean(gs_approx < 0)),
            'gs_weighted'    : float(np.mean(gs_approx)),
            'tone_mean'      : float(np.mean(tone_arr)),
            'tone_std'       : float(np.std(tone_arr, ddof=0)),
            'n_events'       : float(n),
            'mentions_sum'   : float(n * 10),
            '_source'        : 'live',
            '_n_articles'    : n,
            '_n_domains'     : len(domains),
        }

    except Exception:
        return {
            'gs_mean'        : 0.0,
            'gs_std'         : 2.0,
            'gs_conflict_pct': 0.5,
            'gs_weighted'    : 0.0,
            'tone_mean'      : -1.0,
            'tone_std'       : 3.0,
            'n_events'       : 50.0,
            'mentions_sum'   : 500.0,
            '_source'        : 'fallback',
            '_n_articles'    : 0,
            '_n_domains'     : 0,
        }


# ── BUILD FEATURE VECTOR ─────────────────────────────────
def build_feature_vector(market_row, gdelt):
    features = {
        'ovx_close'      : market_row.get('ovx_close',       np.nan),
        'vix_close'      : market_row.get('vix_close',       np.nan),
        'oil_vol_5d'     : market_row.get('oil_vol_5d',      np.nan),
        'oil_vol_20d'    : market_row.get('oil_vol_20d',     np.nan),
        'oil_close'      : market_row.get('oil_close',       np.nan),
        'dxy_close'      : market_row.get('dxy_close',       np.nan),
        'gold_oil_ratio' : market_row.get('gold_oil_ratio',  np.nan),
        'gs_mean'        : gdelt['gs_mean'],
        'gs_std'         : gdelt['gs_std'],
        'gs_conflict_pct': gdelt['gs_conflict_pct'],
        'gs_weighted'    : gdelt['gs_weighted'],
        'tone_mean'      : gdelt['tone_mean'],
        'tone_std'       : gdelt['tone_std'],
        'n_events'       : gdelt['n_events'],
        'mentions_sum'   : gdelt['mentions_sum'],
    }
    return np.array([[features[c] for c in FEATURE_COLS]], dtype=np.float32)


# ── COMPUTE REALIZED 4H VOL ──────────────────────────────
def compute_realized_4h_vol(market_df, prediction_time):
    now = datetime.now()
    if now < prediction_time + timedelta(hours=4):
        return None

    try:
        start = (prediction_time - timedelta(hours=1)).strftime('%Y-%m-%d')
        end   = (prediction_time + timedelta(hours=6)).strftime('%Y-%m-%d')
        h = yf.download('BZ=F', start=start, end=end,
                        interval='1h', progress=False, auto_adjust=True)
        if h.empty or len(h) < 4:
            return None

        if isinstance(h.columns, pd.MultiIndex):
            h.columns = [c[0] for c in h.columns]

        h.index = pd.to_datetime(h.index).tz_localize(None)
        h = h.sort_index()

        mask  = (h.index >= prediction_time) & \
                (h.index <= prediction_time + timedelta(hours=4))
        h4    = h[mask]

        if len(h4) < 2:
            return None

        returns = np.log(h4['Close'] / h4['Close'].shift(1)).dropna()
        rvol    = float(returns.std(ddof=1) * np.sqrt(252 * 23))
        return rvol

    except Exception:
        return None


# ── PREDICTIONS LOG ──────────────────────────────────────
def load_log():
    if os.path.exists(PREDICTIONS_LOG):
        return pd.read_csv(PREDICTIONS_LOG, parse_dates=['timestamp'])
    return pd.DataFrame(columns=[
        'timestamp', 'maml_pred', 'mlp_pred',
        'oil_close', 'ovx_close', 'actual_rvol_4h'
    ])


def save_log(df):
    df.to_csv(PREDICTIONS_LOG, index=False)


def append_prediction(timestamp, maml_pred, mlp_pred, oil_close, ovx_close):
    log = load_log()
    new_row = pd.DataFrame([{
        'timestamp'      : timestamp,
        'maml_pred'      : maml_pred,
        'mlp_pred'       : mlp_pred,
        'oil_close'      : oil_close,
        'ovx_close'      : ovx_close,
        'actual_rvol_4h' : np.nan,
    }])
    log = pd.concat([log, new_row], ignore_index=True)
    save_log(log)


def update_actuals(market_df):
    log = load_log()
    if log.empty:
        return log

    now = datetime.now()
    for idx, row in log.iterrows():
        if pd.isna(row['actual_rvol_4h']):
            pred_time = pd.to_datetime(row['timestamp'])
            if now >= pred_time + timedelta(hours=4):
                actual = compute_realized_4h_vol(market_df, pred_time)
                if actual is not None:
                    log.at[idx, 'actual_rvol_4h'] = actual

    save_log(log)
    return log


# ══════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════
st.title("Oil Volatility Forecaster")
st.caption("MAML-adapted MLP · GDELT geopolitical events + market data · 4h forward vol")

# Auto-refresh every 15 minutes
st.markdown(
    '<meta http-equiv="refresh" content="900">',
    unsafe_allow_html=True
)

# ── ABSTRACT ─────────────────────────────────────────────
with st.expander("About this dashboard", expanded=True):
    st.markdown("""
    This dashboard uses a **Model-Agnostic Meta-Learning (MAML)** model trained on
    GDELT geopolitical event data and Brent oil market features to forecast
    **4-hour forward oil realized volatility**.

    **How predictions are made:**
    The model fetches live oil-related news from GDELT and current market data
    (Brent crude price, OVX, VIX, DXY, Gold/Oil ratio, rolling volatility).
    The MAML model adapts to recent market conditions using the last 5 trading
    days as a support set, then outputs a predicted annualized 4h volatility.

    **How actuals are computed:**
    After 4 hours have elapsed since a prediction was logged, the dashboard
    fetches hourly Brent crude prices and computes the standard deviation of
    returns over that 4-hour window, annualized with √(252 × 23).

    **Label convention used throughout this dashboard:**
    - **(Predicted)** — values output by the model before the fact
    - **(Actual)** — values computed from real market data after the fact

    Click **Make Prediction Now** in the sidebar to log a prediction and
    begin tracking. Actuals will appear automatically after 4 hours.
    """)
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    gdelt_hours   = st.slider("GDELT lookback (hours)", 1, 72, 24)
    make_pred_btn = st.button("Make Prediction Now", type="primary")
    clear_log_btn = st.button("Clear Log")
    if clear_log_btn:
        if os.path.exists(PREDICTIONS_LOG):
            os.remove(PREDICTIONS_LOG)
        st.success("Log cleared.")

    st.markdown("---")
    st.markdown("**Files loaded:**")
    st.markdown("- `maml_trained.pth`")
    st.markdown("- `mlp_pretrained.pth`")
    st.markdown("- `feature_scaler.pkl`")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown(
        "1. Fetches GDELT oil events\n"
        "2. Fetches live market data\n"
        "3. MAML adapts on last 5 days\n"
        "4. Predicts 4h forward vol\n"
        "5. After 4h, computes actual vol\n"
        "6. Plots actual vs predicted"
    )

# ── Fetch data ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Market Data")
    with st.spinner("Fetching from Yahoo Finance..."):
        market = fetch_market_data(days_back=60)

    if market.empty:
        st.error("Could not fetch market data.")
        st.stop()

    latest = market.iloc[-1].to_dict()
    m1, m2, m3 = st.columns(3)
    m1.metric("Brent Crude (Actual)", f"${latest.get('oil_close', 0):.2f}")
    m2.metric("OVX (Actual)", f"{latest.get('ovx_close', 0):.2f}")
    m3.metric("VIX (Actual)", f"{latest.get('vix_close', 0):.2f}")
    st.caption("All market values are actual live data from Yahoo Finance")
    st.dataframe(market.tail(5).round(4), use_container_width=True)

with col2:
    st.subheader("GDELT Events")
    with st.spinner("Fetching from GDELT..."):
        gdelt = fetch_gdelt_events(hours_back=gdelt_hours)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Events (Actual)",    f"{int(gdelt['n_events'])}")
    g2.metric("Sources (Actual)",   f"{int(gdelt['mentions_sum'] / max(gdelt['n_events'],1)):.0f}")
    g3.metric("Avg Tone (Actual)",  f"{gdelt['tone_mean']:.2f}")
    g4.metric("Conflict% (Actual)", f"{gdelt['gs_conflict_pct']*100:.0f}%")

    # CHANGE 2: show live vs fallback status
    if gdelt.get('_source') == 'live':
        st.success(f"Live GDELT data — {gdelt['_n_articles']} articles "
                   f"from {gdelt['_n_domains']} sources in last {gdelt_hours}h")
    else:
        st.warning("GDELT API unavailable — using neutral fallback values. "
                   "Predictions will rely more heavily on market features.")

# ── Make prediction ──────────────────────────────────────
st.markdown("---")
st.subheader("Volatility Forecast")

raw_X = build_feature_vector(latest, gdelt)

if np.isnan(raw_X).any():
    missing = [FEATURE_COLS[i] for i in range(15) if np.isnan(raw_X[0][i])]
    st.error(f"Missing features: {missing}")
    st.stop()

scaled_X = scaler.transform(raw_X)
X_tensor = torch.tensor(scaled_X, dtype=torch.float32)

# Plain MLP prediction
plain_mlp.eval()
with torch.no_grad():
    mlp_pred_log = plain_mlp(X_tensor).item()
    mlp_pred     = float(np.expm1(mlp_pred_log))

# MAML adaptation using last 5 days as support set
support_rows = market.tail(5)
sup_features = []
sup_vols     = []

for _, row in support_rows.iterrows():
    f = build_feature_vector(row.to_dict(), gdelt)
    if not np.isnan(f).any():
        sup_features.append(scaler.transform(f)[0])
        if not np.isnan(row.get('oil_vol_5d', np.nan)):
            sup_vols.append(row['oil_vol_5d'])

if len(sup_features) >= 2 and len(sup_vols) >= 2:
    n    = min(len(sup_features), len(sup_vols))
    sup_X = torch.tensor(
        np.array(sup_features[:n], dtype=np.float32)
    )
    sup_y = torch.tensor(
        np.log1p(np.array(sup_vols[:n], dtype=np.float32))
    ).unsqueeze(1)
    adapted = adapt(maml_model, sup_X, sup_y)
else:
    adapted = maml_model

adapted.eval()
with torch.no_grad():
    maml_pred_log = adapted(X_tensor).item()
    maml_pred     = float(np.expm1(maml_pred_log))

# Display
st.markdown("Values below are **model predictions** — not yet verified against actuals")
p1, p2, p3 = st.columns(3)
p1.metric("Plain MLP (Predicted 4h vol)", f"{mlp_pred:.4f}")
p2.metric("MAML (Predicted 4h vol)",
          f"{maml_pred:.4f}",
          delta=f"{maml_pred - mlp_pred:+.4f} vs MLP")
vol_label = "LOW" if maml_pred < 0.15 else \
            "MODERATE" if maml_pred < 0.30 else "HIGH"
p3.metric("Vol Regime (Predicted)", vol_label)
st.caption("MAML adapts on the last 5 trading days before predicting. "
           "Actual vol will be computed from hourly Brent prices after 4 hours.")

# Log prediction when button pressed
now = datetime.now()
if make_pred_btn:
    append_prediction(
        timestamp  = now,
        maml_pred  = maml_pred,
        mlp_pred   = mlp_pred,
        oil_close  = latest.get('oil_close', np.nan),
        ovx_close  = latest.get('ovx_close', np.nan),
    )
    st.success(f"Prediction logged at {now.strftime('%H:%M:%S')}. "
               f"Actual vol will be computed in 4 hours.")

# ── Update actuals and plot ───────────────────────────────
st.markdown("---")
st.subheader("Actual vs Predicted (4h Realized Vol)")

with st.spinner("Checking for resolved predictions..."):
    log = update_actuals(market)

if log.empty:
    st.info("No predictions logged yet. Click 'Make Prediction Now' to start tracking.")
else:
    resolved = log.dropna(subset=['actual_rvol_4h'])
    pending  = log[log['actual_rvol_4h'].isna()]

    if not resolved.empty:
        plot_df = resolved[['timestamp', 'maml_pred', 'mlp_pred', 'actual_rvol_4h']].copy()
        plot_df = plot_df.set_index('timestamp').sort_index()
        plot_df.columns = [
            'MAML (Predicted)',
            'MLP (Predicted)',
            'Realized Vol (Actual)'
        ]
        st.markdown("**Predicted** = model forecast at time of logging  |  "
                    "**Actual** = realized vol computed 4h later from hourly Brent prices")
        st.line_chart(plot_df, use_container_width=True)

        mae_maml = float(np.mean(np.abs(
            resolved['actual_rvol_4h'] - resolved['maml_pred']
        )))
        mae_mlp = float(np.mean(np.abs(
            resolved['actual_rvol_4h'] - resolved['mlp_pred']
        )))

        r1, r2, r3 = st.columns(3)
        r1.metric("Resolved predictions", len(resolved))
        r2.metric("MAML MAE (Predicted vs Actual)", f"{mae_maml:.4f}")
        r3.metric("MLP MAE (Predicted vs Actual)",  f"{mae_mlp:.4f}")

        display_df = resolved[[
            'timestamp', 'maml_pred', 'mlp_pred', 'actual_rvol_4h',
            'oil_close', 'ovx_close'
        ]].copy()
        display_df.columns = [
            'Timestamp',
            'MAML (Predicted)',
            'MLP (Predicted)',
            'Realized Vol (Actual)',
            'Oil Close (Actual)',
            'OVX (Actual)',
        ]
        st.dataframe(
            display_df.sort_values('Timestamp', ascending=False).round(5),
            use_container_width=True
        )
    else:
        st.info("Predictions made but 4 hours haven't passed yet. Check back later.")

    if not pending.empty:
        st.markdown(f"**⏳ {len(pending)} prediction(s) pending actuals** — check back in 4 hours")
        pending_display = pending[['timestamp', 'maml_pred', 'mlp_pred', 'oil_close', 'ovx_close']].copy()
        pending_display.columns = [
            'Timestamp',
            'MAML (Predicted)',
            'MLP (Predicted)',
            'Oil Close (Actual at prediction time)',
            'OVX (Actual at prediction time)',
        ]
        now_ts = datetime.now()
        pending_display['Time until actual'] = pending_display['Timestamp'].apply(
            lambda t: f"{max(0, int((t + timedelta(hours=4) - now_ts).total_seconds() / 60))} min remaining"
        )
        st.dataframe(
            pending_display.sort_values('Timestamp', ascending=False).round(5),
            use_container_width=True
        )

# ── Historical charts ─────────────────────────────────────
st.markdown("---")
st.subheader("Market Charts")

tab1, tab2, tab3 = st.tabs(["Oil Price (Actual)", "Realized Vol (Actual)", "VIX / OVX (Actual)"])
with tab1:
    if 'oil_close' in market.columns:
        st.line_chart(market['oil_close'], use_container_width=True)
with tab2:
    vol_cols = [c for c in ['oil_vol_5d', 'oil_vol_20d'] if c in market.columns]
    if vol_cols:
        st.line_chart(market[vol_cols], use_container_width=True)
with tab3:
    ind_cols = [c for c in ['vix_close', 'ovx_close'] if c in market.columns]
    if ind_cols:
        st.line_chart(market[ind_cols], use_container_width=True)

# ── Debug panel ───────────────────────────────────────────
with st.expander("Feature Vector (debug)"):
    st.dataframe(pd.DataFrame({
        'Feature'      : FEATURE_COLS,
        'Raw Value'    : raw_X[0],
        'Scaled Value' : scaled_X[0],
    }), use_container_width=True)

st.markdown("---")
st.caption(
    f"Last refresh: {now.strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Auto-refresh: every 15 min | "
    f"Data: Yahoo Finance + GDELT 2.0"
)