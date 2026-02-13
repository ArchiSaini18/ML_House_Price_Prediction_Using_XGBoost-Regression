import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
import warnings
warnings.simplefilter("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏡",
    layout="wide",
)

# ── Train Model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    dataset = fetch_california_housing()
    house_price = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    house_price["Target"] = dataset.target
    X = house_price.drop("Target", axis=1)
    Y = house_price["Target"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(x_train, y_train)
    train_r2  = metrics.r2_score(y_train, model.predict(x_train))
    test_r2   = metrics.r2_score(y_test,  model.predict(x_test))
    test_mse  = metrics.mean_squared_error(y_test, model.predict(x_test))
    return model, train_r2, test_r2, test_mse, len(X)

with st.spinner("Initialising model…"):
    model, train_r2, test_r2, test_mse, n_samples = load_model()

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  —  Golden Amber · Dark Theme · matching diabetes.py style
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

/* ── Variables ── */
:root {
    --bg:           #090b0f;
    --surface:      #0f1318;
    --surface2:     #141a22;
    --border:       #1c2535;
    --border-hi:    #263245;
    --amber:        #f59e0b;
    --amber-light:  #fde68a;
    --amber-dim:    rgba(245,158,11,0.13);
    --sky:          #38bdf8;
    --red:          #f43f5e;
    --green:        #10b981;
    --text:         #f0ead8;
    --muted:        #5a6070;
    --head-font:    'Playfair Display', Georgia, serif;
    --mono-font:    'IBM Plex Mono', monospace;
    --glow-amber:   0 0 28px rgba(245,158,11,0.22);
}

/* ── Global ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: var(--bg) !important;
    font-family: var(--mono-font) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }

/* ══════════════════════════════
   HERO
══════════════════════════════ */
.hero {
    position: relative;
    padding: 3rem 3.5rem 2.5rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, #090b0f 0%, #101620 60%, #090b0f 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 90% 50%, rgba(245,158,11,0.09) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 10% 80%, rgba(56,189,248,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--amber), var(--amber-light), var(--sky), transparent);
}
.hero-tag {
    display: inline-block;
    font-family: var(--mono-font);
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--amber);
    background: var(--amber-dim);
    border: 1px solid rgba(245,158,11,0.28);
    border-radius: 4px;
    padding: 0.25rem 0.8rem;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: var(--head-font) !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    line-height: 1.1 !important;
    letter-spacing: -0.01em;
    margin: 0 0 0.6rem 0 !important;
}
.hero h1 span {
    background: linear-gradient(135deg, var(--amber), var(--amber-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 0.04em;
}
.hero-badge {
    position: absolute;
    right: 3.5rem; top: 50%;
    transform: translateY(-50%);
    width: 80px; height: 80px;
    border-radius: 50%;
    background: var(--amber-dim);
    border: 1px solid rgba(245,158,11,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem;
    box-shadow: var(--glow-amber);
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--amber);
    box-shadow: 0 0 6px var(--amber);
    margin-right: 0.5rem;
    vertical-align: middle;
}
.status-badge {
    font-size: 0.7rem; letter-spacing: 0.1em;
    color: var(--muted); text-transform: uppercase;
}

/* ── Metric Chips ── */
.acc-row {
    display: flex; gap: 0.8rem; margin-bottom: 2.5rem; flex-wrap: wrap;
}
.acc-chip {
    background: var(--surface2);
    border: 1px solid var(--border-hi);
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
.acc-chip span {
    font-family: var(--head-font);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--amber);
    margin-right: 0.35rem;
}

/* ══════════════════════════════
   SECTION LABELS
══════════════════════════════ */
.section-label {
    font-family: var(--mono-font);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--amber);
    border-left: 2px solid var(--amber);
    padding-left: 0.7rem;
    margin-bottom: 1.4rem;
    margin-top: 0.5rem;
}

/* ══════════════════════════════
   FORM PANELS
══════════════════════════════ */
.form-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.8rem 2rem 2rem;
    position: relative;
    margin-bottom: 1.4rem;
    transition: border-color 0.3s;
}
.form-panel:hover { border-color: var(--border-hi); }
.form-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 1.5rem; right: 1.5rem; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(245,158,11,0.15), transparent);
}

/* ══════════════════════════════
   INPUTS
══════════════════════════════ */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    font-family: var(--mono-font) !important;
    font-size: 0.74rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 0.25rem !important;
}
[data-testid="stNumberInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono-font) !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 0.85rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.12) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] button {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--muted) !important;
}
[data-testid="stNumberInput"] button:hover {
    background: var(--amber-dim) !important;
    color: var(--amber) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono-font) !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.12) !important;
}
[data-testid="stSelectbox"] svg { color: var(--amber) !important; }
[data-testid="stSelectbox"] ul {
    background: var(--surface2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] li {
    font-family: var(--mono-font) !important;
    font-size: 0.85rem !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] li:hover {
    background: var(--amber-dim) !important;
    color: var(--amber-light) !important;
}

/* ══════════════════════════════
   BUTTON
══════════════════════════════ */
[data-testid="stButton"] > button {
    width: 100% !important;
    height: 58px !important;
    background: linear-gradient(135deg, #b45309, var(--amber), #fde68a) !important;
    color: #0a0800 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--head-font) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(245,158,11,0.30), 0 1px 0 rgba(255,255,255,0.08) inset !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(245,158,11,0.45), 0 2px 0 rgba(255,255,255,0.12) inset !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ══════════════════════════════
   RESULT CARDS
══════════════════════════════ */
.result-wrap {
    animation: fadeSlideUp 0.5s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* High price */
.result-high {
    background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(9,11,15,0) 60%);
    border: 1px solid rgba(245,158,11,0.38);
    border-top: 3px solid var(--amber);
    border-radius: 14px;
    padding: 2.5rem 2.5rem 2rem;
    position: relative; overflow: hidden;
}
.result-high::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 60% at 90% 10%, rgba(245,158,11,0.08) 0%, transparent 70%);
    pointer-events: none;
}

/* Mid price */
.result-mid {
    background: linear-gradient(135deg, rgba(56,189,248,0.10) 0%, rgba(9,11,15,0) 60%);
    border: 1px solid rgba(56,189,248,0.32);
    border-top: 3px solid var(--sky);
    border-radius: 14px;
    padding: 2.5rem 2.5rem 2rem;
    position: relative; overflow: hidden;
}
.result-mid::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 60% at 90% 10%, rgba(56,189,248,0.07) 0%, transparent 70%);
    pointer-events: none;
}

/* Low price */
.result-low {
    background: linear-gradient(135deg, rgba(16,185,129,0.10) 0%, rgba(9,11,15,0) 60%);
    border: 1px solid rgba(16,185,129,0.32);
    border-top: 3px solid var(--green);
    border-radius: 14px;
    padding: 2.5rem 2.5rem 2rem;
    position: relative; overflow: hidden;
}
.result-low::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 60% at 90% 10%, rgba(16,185,129,0.07) 0%, transparent 70%);
    pointer-events: none;
}

.result-icon  { font-size: 3rem; margin-bottom: 1rem; display: block; }
.result-verdict {
    font-family: var(--head-font);
    font-size: 1.3rem; font-weight: 600;
    margin-bottom: 0.4rem; line-height: 1.2;
    color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em;
}
.result-price {
    font-family: var(--head-font);
    font-size: 3rem; font-weight: 700;
    line-height: 1; margin-bottom: 0.5rem;
}
.result-price.amber { color: var(--amber); }
.result-price.sky   { color: var(--sky); }
.result-price.green { color: var(--green); }

.result-sub {
    font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem;
}

/* Price breakdown bar */
.price-range-wrap { margin-top: 1.6rem; }
.price-range-label {
    display: flex; justify-content: space-between;
    font-size: 0.7rem; color: var(--muted);
    letter-spacing: 0.08em; margin-bottom: 0.4rem;
}
.price-range-track {
    width: 100%; height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    position: relative; overflow: visible;
}
.price-range-fill {
    position: absolute; left: 0; top: 0;
    height: 100%; border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.22,1,0.36,1);
}
.price-range-dot {
    position: absolute; top: 50%;
    transform: translate(-50%, -50%);
    width: 14px; height: 14px;
    border-radius: 50%;
    border: 2px solid var(--bg);
    box-shadow: 0 0 8px currentColor;
}

.result-note {
    font-size: 0.78rem; color: var(--muted);
    margin-top: 1.4rem; line-height: 1.65;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 1rem;
}

/* Feature insight mini cards */
.insight-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.6rem; margin-top: 1.4rem;
}
.insight-chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-size: 0.72rem;
}
.insight-chip .ic-label {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}
.insight-chip .ic-value {
    font-family: var(--head-font);
    font-size: 1rem; font-weight: 600;
    color: var(--text);
}

/* Idle card */
.idle-card {
    background: var(--surface);
    border: 1px dashed var(--border-hi);
    border-radius: 14px;
    padding: 3.5rem 2rem;
    text-align: center; color: var(--muted);
}
.idle-icon { font-size: 2.8rem; margin-bottom: 1rem; opacity: 0.45; }
.idle-head {
    font-family: var(--head-font);
    font-size: 1.2rem; color: #2a2a1a; margin-bottom: 0.4rem;
}
.idle-body { font-size: 0.8rem; line-height: 1.6; }

/* ── Layout helpers ── */
[data-testid="stHorizontalBlock"] {
    gap: 1.4rem !important;
    align-items: stretch !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-tag">⬡ Real Estate Intelligence Platform</div>
    <h1>California House<br><span>Price</span> Prediction</h1>
    <p class="hero-sub">
        XGBoost Regressor · California Housing Dataset &nbsp;·&nbsp;
        <span class="status-dot"></span>
        <span class="status-badge">Model Active</span>
    </p>
    <div class="hero-badge">🏡</div>
</div>
<div class="acc-row">
    <div class="acc-chip"><span>{train_r2*100:.1f}%</span> Train R² Score</div>
    <div class="acc-chip"><span>{test_r2*100:.1f}%</span> Test R² Score</div>
    <div class="acc-chip"><span>{n_samples:,}</span> Training Samples</div>
    <div class="acc-chip"><span>8</span> Input Features</div>
    <div class="acc-chip"><span>${test_mse*100000:.0f}</span> RMSE (×$100K)</div>
</div>
""", unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
form_col, result_col = st.columns([1.05, 0.95], gap="medium")

with form_col:

    # ── Section 01 : Household Demographics ──────────────────────────────────
    st.markdown('<div class="section-label">01 — Household Demographics</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-panel">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        MedInc      = st.number_input("Median Income (×$10K)",      0.5,  15.0,  3.87, step=0.01, format="%.4f",
                                       help="Median household income in tens of thousands of dollars")
        HouseAge    = st.number_input("House Age (years)",           1.0,  52.0, 28.64, step=1.0,  format="%.1f",
                                       help="Median age of houses in the block")
    with c2:
        Population  = st.number_input("Block Population",           3.0, 35682.0, 1425.5, step=1.0, format="%.1f",
                                       help="Total population in the block")
        AveOccup    = st.number_input("Avg Occupancy (persons/house)", 0.5, 1243.0, 3.07, step=0.01, format="%.4f",
                                       help="Average number of household members")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 02 : Property Characteristics ────────────────────────────────
    st.markdown('<div class="section-label">02 — Property Characteristics</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-panel">', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        AveRooms    = st.number_input("Avg Rooms per House",         0.85, 141.9,  5.43, step=0.01, format="%.4f",
                                       help="Average number of rooms per household")
        AveBedrms   = st.number_input("Avg Bedrooms per House",      0.33,  34.07, 1.10, step=0.01, format="%.4f",
                                       help="Average number of bedrooms per household")
    with c4:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.15);
                    border-radius:8px;padding:0.9rem 1rem;margin-top:1.8rem;font-size:0.78rem;
                    color:#6a6040;line-height:1.7">
            💡 <strong style="color:#a07820">Tip:</strong> Avg Bedrooms / Avg Rooms ratio
            (ideally 0.15–0.25) indicates room composition quality.
            Lower ratios often correlate with higher property values.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 03 : Geographic Location ─────────────────────────────────────
    st.markdown('<div class="section-label">03 — Geographic Location</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-panel">', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        Latitude    = st.number_input("Latitude",   32.54,  41.95,  35.63, step=0.01, format="%.4f",
                                       help="Block latitude — Southern CA (~32–34) vs Northern CA (~37–42)")
    with c6:
        Longitude   = st.number_input("Longitude", -124.35, -114.31, -119.57, step=0.01, format="%.4f",
                                       help="Block longitude — Coastal (-124 to -120) vs Inland (-120 to -114)")

    # Quick location hint
    region = ""
    if Latitude > 37.5:
        region = "🌉 Northern California (Bay Area / Sacramento)"
    elif Latitude > 34.5:
        region = "🌴 Central California (Fresno / Central Valley)"
    else:
        region = "☀️ Southern California (LA / San Diego)"

    st.markdown(f"""
    <div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.15);
                border-radius:8px;padding:0.7rem 1rem;margin-top:0.3rem;font-size:0.8rem;
                color:#3a6070;letter-spacing:0.04em">
        📍 Detected Region: <strong style="color:#38bdf8">{region}</strong>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Estimate House Price →")

# ── Result Column ─────────────────────────────────────────────────────────────
with result_col:
    st.markdown('<div class="section-label">04 — Price Estimate</div>', unsafe_allow_html=True)

    if predict_btn:
        input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                                 Population, AveOccup, Latitude, Longitude]])
        predicted  = model.predict(input_data)[0]          # value in $100K units
        price_usd  = predicted * 100_000                   # convert to dollars

        # Classify tier
        if price_usd >= 350_000:
            tier = "high"
            tier_label = "Premium Market"
            icon = "🏆"
            bar_color_css = "background: linear-gradient(90deg, #b45309, #f59e0b); box-shadow: 0 0 10px rgba(245,158,11,0.5);"
            pct_class = "amber"
            card_class = "result-high"
        elif price_usd >= 180_000:
            tier = "mid"
            tier_label = "Mid-Range Market"
            icon = "🏠"
            bar_color_css = "background: linear-gradient(90deg, #0369a1, #38bdf8); box-shadow: 0 0 10px rgba(56,189,248,0.5);"
            pct_class = "sky"
            card_class = "result-mid"
        else:
            tier = "low"
            tier_label = "Affordable Market"
            icon = "🏘️"
            bar_color_css = "background: linear-gradient(90deg, #065f46, #10b981); box-shadow: 0 0 10px rgba(16,185,129,0.5);"
            pct_class = "green"
            card_class = "result-low"

        # Format price
        if price_usd >= 1_000_000:
            price_str = f"${price_usd/1_000_000:.2f}M"
        else:
            price_str = f"${price_usd:,.0f}"

        st.markdown(f"""
        <div class="result-wrap">
        <div class="{card_class}">
            <span class="result-icon">{icon}</span>
            <div class="result-verdict">{tier_label}</div>
            <div class="result-price {pct_class}">{price_str}</div>
            <div class="result-sub">Predicted median house value</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="idle-card">
            <div class="idle-icon">◈</div>
            <div class="idle-head">Awaiting Estimate</div>
            <div class="idle-body">
                Fill in the property details across<br>
                the three sections on the left,<br>
                then click <em>Estimate House Price</em>.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2rem;font-size:0.72rem;color:#3a3010;
                border-top:1px solid #1c2535;padding-top:1rem;line-height:1.7">
        Predictions are based on the 1990 California Census and are for
        educational purposes only. Actual market values may differ significantly.
        Consult a licensed real estate professional for accurate valuations.
    </div>
    """, unsafe_allow_html=True)