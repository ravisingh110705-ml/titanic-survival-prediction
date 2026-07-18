import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
#  LOAD FONTS
# ──────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a26;
        --bg-card-hover: #22223a;
        --gold: #d4a853;
        --gold-light: #f0d48a;
        --gold-dark: #a07830;
        --text-primary: #f0ece4;
        --text-secondary: #8a8698;
        --text-muted: #5a566a;
        --border: #2a2a3e;
        --success: #4ade80;
        --danger: #f87171;
    }

    html, body, [class*="st-"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── FIX: restore Streamlit's own icon font AFTER the broad override above.
       This is what was broken — the [class*="st-"] rule above was forcing
       Inter font onto icon elements too, so icons like the sidebar arrow
       showed as raw text ("keyboard_double_arrow_right") instead of a glyph. ── */
    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"],
    [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 24px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: rgba(10,10,15,0.85) !important; backdrop-filter: blur(20px); }
    .stApp { border-top: none !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e0e16 0%, #14141f 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--gold-dark), var(--gold), var(--gold-dark));
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--gold) !important;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
        display: block;
    }

    /* ── Number Input buttons ── */
    [data-testid="stNumberInput"] button,
    .stNumberInput button {
        color: transparent !important;
        font-size: 0 !important;
        line-height: 0 !important;
        text-indent: -9999px !important;
        overflow: hidden !important;
        position: relative !important;
        width: 32px !important;
        min-width: 32px !important;
        height: 32px !important;
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        padding: 0 !important;
    }
    [data-testid="stNumberInput"] button span,
    .stNumberInput button span {
        color: transparent !important;
        font-size: 0 !important;
        display: none !important;
    }
    [data-testid="stNumberInput"] button:hover,
    .stNumberInput button:hover {
        background: var(--gold-dark) !important;
        border-color: var(--gold) !important;
    }
    [data-testid="stNumberInput"] button:first-of-type::after,
    .stNumberInput button:first-of-type::after {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: 0; height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-bottom: 6px solid var(--gold);
    }
    [data-testid="stNumberInput"] button:last-of-type::after,
    .stNumberInput button:last-of-type::after {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: 0; height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid var(--gold);
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        transition: border-color 0.3s, box-shadow 0.3s !important;
    }
    .stSelectbox > div > div:hover,
    .stSelectbox > div > div:focus-within {
        border-color: var(--gold-dark) !important;
        box-shadow: 0 0 0 3px rgba(212,168,83,0.1) !important;
    }
    .stSelectbox label {
        color: var(--text-secondary) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="select"] span {
        color: var(--text-primary) !important;
    }

    /* ── Slider ── */
    .stSlider > div > div > div > div {
        background: var(--bg-card) !important;
        border-radius: 10px !important;
    }
    .stSlider [data-testid="stWidgetLabel"] p {
        color: var(--text-secondary) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    .stSlider [data-testid="stWidgetLabel"] div {
        color: var(--gold) !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    [data-baseweb="slider"] [role="slider"] {
        background: var(--gold) !important;
        box-shadow: 0 0 10px rgba(212,168,83,0.4) !important;
    }
    [data-baseweb="slider"] [class*="track"] { background: var(--border) !important; }
    [data-baseweb="slider"] [class*="fill"] {
        background: linear-gradient(90deg, var(--gold-dark), var(--gold)) !important;
    }

    /* ── Number Input field ── */
    .stNumberInput > div > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        transition: border-color 0.3s, box-shadow 0.3s !important;
    }
    .stNumberInput > div > div > div:hover,
    .stNumberInput > div > div > div:focus-within {
        border-color: var(--gold-dark) !important;
        box-shadow: 0 0 0 3px rgba(212,168,83,0.1) !important;
    }
    .stNumberInput label {
        color: var(--text-secondary) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    .stNumberInput [data-testid="stWidgetLabel"] div { color: var(--gold) !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--gold-dark), var(--gold), var(--gold-light)) !important;
        color: #0a0a0f !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.04em !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(212,168,83,0.25) !important;
        width: 100%;
        text-transform: uppercase;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(212,168,83,0.4) !important;
        background: linear-gradient(135deg, var(--gold), var(--gold-light), #fff5d4) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 10px rgba(212,168,83,0.2) !important;
    }

    /* ── Hero Section ── */
    .hero-section {
        position: relative;
        padding: 3rem 2.5rem 2.5rem;
        margin-bottom: 1.5rem;
        border-radius: 20px;
        overflow: hidden;
        background: linear-gradient(135deg, #12121a 0%, #1a1520 50%, #161822 100%);
        border: 1px solid var(--border);
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(212,168,83,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-section::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--gold-dark), var(--gold), var(--gold-dark), transparent);
    }
    .hero-title {
        font-family: 'Playfair Display', serif !important;
        font-weight: 900 !important;
        font-size: 2.8rem !important;
        line-height: 1.1 !important;
        color: var(--text-primary) !important;
        margin-bottom: 0.75rem !important;
    }
    .hero-title .gold { color: var(--gold) !important; }
    .hero-subtitle {
        font-size: 1.05rem !important;
        color: var(--text-secondary) !important;
        font-weight: 300 !important;
        line-height: 1.6 !important;
        max-width: 600px;
    }

    /* ── Stat Cards ── */
    .stat-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 14px; padding: 1.25rem 1.5rem;
        transition: all 0.3s ease; position: relative; overflow: hidden;
    }
    .stat-card:hover {
        border-color: var(--gold-dark); transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .stat-card .stat-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .stat-card .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem; font-weight: 700; color: var(--text-primary);
    }
    .stat-card .stat-label {
        font-size: 0.75rem; color: var(--text-muted);
        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem;
    }

    /* ── Result Card ── */
    .result-card {
        border-radius: 20px; padding: 2.5rem;
        position: relative; overflow: hidden;
        animation: fadeSlideUp 0.6s ease-out;
    }
    .result-card.survived {
        background: linear-gradient(135deg, rgba(74,222,128,0.06), rgba(74,222,128,0.02));
        border: 1px solid rgba(74,222,128,0.2);
    }
    .result-card.not-survived {
        background: linear-gradient(135deg, rgba(248,113,113,0.06), rgba(248,113,113,0.02));
        border: 1px solid rgba(248,113,113,0.2);
    }
    .result-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
    .result-card.survived::before { background: linear-gradient(90deg, transparent, var(--success), transparent); }
    .result-card.not-survived::before { background: linear-gradient(90deg, transparent, var(--danger), transparent); }
    .result-emoji { font-size: 3.5rem; margin-bottom: 0.75rem; display: block; animation: pulse 2s ease-in-out infinite; }
    .result-title { font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-card.survived .result-title { color: var(--success); }
    .result-card.not-survived .result-title { color: var(--danger); }
    .result-subtitle { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem; }

    .prob-bar-container { background: var(--bg-primary); border-radius: 10px; height: 14px; overflow: hidden; position: relative; margin: 0.5rem 0; }
    .prob-bar-fill { height: 100%; border-radius: 10px; transition: width 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94); position: relative; }
    .prob-bar-fill.survived { background: linear-gradient(90deg, #16a34a, var(--success), #86efac); box-shadow: 0 0 15px rgba(74,222,128,0.3); }
    .prob-bar-fill.not-survived { background: linear-gradient(90deg, #dc2626, var(--danger), #fca5a5); box-shadow: 0 0 15px rgba(248,113,113,0.3); }
    .prob-label { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.35rem; font-weight: 500; }

    .insight-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.25rem; transition: all 0.3s ease; }
    .insight-card:hover { border-color: rgba(212,168,83,0.3); background: var(--bg-card-hover); }
    .insight-icon { font-size: 1.3rem; margin-bottom: 0.4rem; display: block; }
    .insight-title { font-size: 0.8rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.2rem; }
    .insight-text { font-size: 0.75rem; color: var(--text-secondary); line-height: 1.5; }

    .passenger-summary { background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem; animation: fadeSlideUp 0.5s ease-out; }
    .summary-row { display: flex; justify-content: space-between; align-items: center; padding: 0.55rem 0; border-bottom: 1px solid rgba(42,42,62,0.5); }
    .summary-row:last-child { border-bottom: none; }
    .summary-key { font-size: 0.82rem; color: var(--text-muted); font-weight: 500; }
    .summary-val { font-size: 0.88rem; color: var(--text-primary); font-weight: 600; }

    .footer-bar { text-align: center; padding: 2rem 0 1rem; color: var(--text-muted); font-size: 0.75rem; border-top: 1px solid var(--border); margin-top: 3rem; }
    .footer-bar .gold { color: var(--gold-dark); }

    @keyframes fadeSlideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.08); } }

    .stSuccess, .stError { display: none !important; }
    .stProgress > div > div > div { background: linear-gradient(90deg, var(--gold-dark), var(--gold)) !important; border-radius: 10px !important; }
    .stProgress > div > div { background: var(--border) !important; border-radius: 10px !important; height: 8px !important; }
    hr { border-color: var(--border) !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--gold-dark); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  LOAD MODEL
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('titanic_pipeline.pkl')

pipeline = load_model()

# ──────────────────────────────────────────────
#  HELPER FUNCTIONS
# ──────────────────────────────────────────────
def get_class_label(pclass):
    return {1: "First Class", 2: "Second Class", 3: "Third Class"}.get(pclass, str(pclass))

def get_embarked_label(emb):
    return {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}.get(emb, emb)

def get_sex_emoji(sex):
    return "👤" if sex == "male" else "👩"

def get_class_emoji(pclass):
    return {1: "👑", 2: "🎩", 3: "🎒"}.get(pclass, "🚶")

def get_insights(pclass, sex, age, sibsp, parch, fare):
    insights = []
    if sex == "female":
        insights.append(("🛟", "Gender Advantage", "Women had significantly higher survival rates due to 'women and children first' protocol."))
    else:
        insights.append(("⚠️", "Gender Factor", "Male passengers faced considerably lower survival odds on the Titanic."))
    if pclass == 1:
        insights.append(("💎", "First Class Privilege", "First-class cabins were on upper decks, closer to lifeboats — a critical survival advantage."))
    elif pclass == 2:
        insights.append(("🏨", "Second Class", "Second-class passengers had moderate survival rates, better than third class but below first."))
    else:
        insights.append(("🔻", "Third Class Challenge", "Third-class passengers were on lower decks with limited access to lifeboats."))
    if age <= 12:
        insights.append(("👶", "Child Priority", "Children were given priority in lifeboat boarding under maritime protocol."))
    elif age >= 60:
        insights.append(("🧓", "Elderly Risk", "Elderly passengers had lower survival rates due to reduced mobility."))
    else:
        insights.append(("🧑", "Adult Age Range", "Being a working-age adult had neutral-to-negative impact on survival chances."))
    total_family = sibsp + parch
    if total_family == 0:
        insights.append(("🧍", "Traveling Alone", "Solo travelers had lower survival rates — family groups helped each other reach lifeboats."))
    elif total_family <= 3:
        insights.append(("👨‍👩‍👧", "Small Family", "Small family groups had good survival dynamics — easier to stay together."))
    else:
        insights.append(("👨‍👩‍👧‍👦", "Large Family", "Large families struggled to stay together, which sometimes reduced individual survival chances."))
    if fare > 100:
        insights.append(("💰", "High Fare Payer", "Higher fares correlate with better cabin locations and earlier access to lifeboats."))
    elif fare < 20:
        insights.append(("🎫", "Low Fare Ticket", "Lower-fare passengers were generally in third class with reduced survival prospects."))
    return insights[:4]

# ──────────────────────────────────────────────
#  SIDEBAR — Input Form
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="padding-top: 1rem;"></div>', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Passenger Details</span>', unsafe_allow_html=True)

    pclass = st.selectbox("Ticket Class", [1, 2, 3], index=2, format_func=lambda x: f"{get_class_label(x)}  {get_class_emoji(x)}")
    sex = st.selectbox("Sex", ["male", "female"], format_func=lambda x: f"{x.capitalize()}  {get_sex_emoji(x)}")
    age = st.slider("Age", 0, 80, 28, step=1)
    sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=10, value=0, step=1)
    fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=32.0, step=0.5)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], format_func=lambda x: f"{get_embarked_label(x)}")

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    predict_btn = st.button("⚡  Predict Survival", use_container_width=True)

    st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color: var(--text-muted); font-size: 0.72rem; padding: 0.5rem; border-top: 1px solid var(--border);">
        Model: SVM Classifier (Pipeline)<br>
        Trained on 891 Titanic passengers
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  MAIN AREA
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-title">Titanic <span class="gold">Survival</span><br>Predictor</div>
    <div class="hero-subtitle">
        Using a machine learning model trained on passenger data, predict whether a passenger
        would have survived the sinking of the RMS Titanic on April 15, 1912.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem;">
    <div class="stat-card">
        <div class="stat-icon">👥</div>
        <div class="stat-value">2,224</div>
        <div class="stat-label">Total Passengers</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">✅</div>
        <div class="stat-value">710</div>
        <div class="stat-label">Survivors</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">❌</div>
        <div class="stat-value">1,514</div>
        <div class="stat-label">Perished</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-value">31.9%</div>
        <div class="stat-label">Survival Rate</div>
    </div>
</div>
""", unsafe_allow_html=True)

if predict_btn:
    new_passenger = pd.DataFrame({
        'Pclass': [pclass], 'Sex': [sex], 'Age': [age],
        'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare], 'Embarked': [embarked]
    })

    prediction = pipeline.predict(new_passenger)[0]
    probability = pipeline.predict_proba(new_passenger)[0]
    survive_prob = probability[1] * 100
    die_prob = probability[0] * 100

    survived = prediction == 1
    status_class = "survived" if survived else "not-survived"
    status_emoji = "✅" if survived else "❌"
    status_text = "SURVIVED" if survived else "DID NOT SURVIVE"
    status_desc = (
        "This passenger would have likely been rescued and survived the tragedy."
        if survived else
        "This passenger would have likely perished in the sinking."
    )

    result_col, summary_col = st.columns([1.4, 1])

    with result_col:
        st.markdown(f"""
        <div class="result-card {status_class}">
            <span class="result-emoji">{status_emoji}</span>
            <div class="result-title">{status_text}</div>
            <div class="result-subtitle">{status_desc}</div>
            <div style="margin-bottom: 0.3rem;">
                <span style="font-size: 0.78rem; color: var(--text-muted); font-weight: 500;">SURVIVAL PROBABILITY</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill {status_class}" style="width: {survive_prob:.1f}%;"></div>
            </div>
            <div class="prob-label">
                <span>Survive: {survive_prob:.1f}%</span>
                <span>Perish: {die_prob:.1f}%</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem;">
                <div style="text-align: center; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 12px;">
                    <div style="font-size: 1.6rem; font-weight: 700; font-family: 'Playfair Display', serif; color: {'var(--success)' if survived else 'var(--danger)'};">{survive_prob:.1f}%</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em;">Survive</div>
                </div>
                <div style="text-align: center; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 12px;">
                    <div style="font-size: 1.6rem; font-weight: 700; font-family: 'Playfair Display', serif; color: {'var(--danger)' if not survived else 'var(--text-secondary)'};">{die_prob:.1f}%</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em;">Perish</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with summary_col:
        st.markdown("""
        <div style="font-size: 0.7rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase;
                     color: var(--gold); margin-bottom: 0.75rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);">
            Passenger Profile
        </div>
        """, unsafe_allow_html=True)

        summary_items = [
            ("Class", f"{get_class_label(pclass)} {get_class_emoji(pclass)}"),
            ("Sex", f"{sex.capitalize()} {get_sex_emoji(sex)}"),
            ("Age", f"{age} years"),
            ("Siblings/Spouses", f"{sibsp}"),
            ("Parents/Children", f"{parch}"),
            ("Fare", f"${fare:.2f}"),
            ("Embarkation", get_embarked_label(embarked)),
            ("Family Size", f"{sibsp + parch + 1}"),
        ]
        rows_html = ""
        for key, val in summary_items:
            rows_html += f'<div class="summary-row"><span class="summary-key">{key}</span><span class="summary-val">{val}</span></div>'
        st.markdown(f'<div class="passenger-summary">{rows_html}</div>', unsafe_allow_html=True)

        confidence = max(survive_prob, die_prob)
        conf_label = "High" if confidence > 80 else "Moderate" if confidence > 60 else "Low"
        conf_color = "var(--success)" if confidence > 80 else "var(--gold)" if confidence > 60 else "var(--danger)"
        st.markdown(f"""
        <div style="margin-top: 1rem; padding: 1rem 1.25rem; background: var(--bg-card);
                     border: 1px solid var(--border); border-radius: 12px; text-align: center;">
            <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;
                        letter-spacing: 0.1em; margin-bottom: 0.3rem;">Model Confidence</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: {conf_color};
                        font-family: 'Playfair Display', serif;">{conf_label} ({confidence:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.7rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase;
                 color: var(--gold); margin-bottom: 1rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);">
        Why This Prediction? — Key Factors
    </div>
    """, unsafe_allow_html=True)

    insights = get_insights(pclass, sex, age, sibsp, parch, fare)
    insight_cols = st.columns(len(insights))
    for i, (icon, title, text) in enumerate(insights):
        with insight_cols[i]:
            st.markdown(f"""
            <div class="insight-card">
                <span class="insight-icon">{icon}</span>
                <div class="insight-title">{title}</div>
                <div class="insight-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: var(--bg-card);
                border: 1px dashed var(--border); border-radius: 20px; margin-top: 1rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.6;">🚢</div>
        <div style="font-family: 'Playfair Display', serif; font-size: 1.4rem;
                    color: var(--text-secondary); margin-bottom: 0.5rem;">Awaiting Passenger Data</div>
        <div style="font-size: 0.88rem; color: var(--text-muted); max-width: 400px; margin: 0 auto; line-height: 1.6;">
            Fill in the passenger details in the sidebar and click
            <span style="color: var(--gold); font-weight: 600;">Predict Survival</span>
            to see the outcome.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div class="insight-card">
            <span class="insight-icon">🕰️</span>
            <div class="insight-title">April 15, 1912</div>
            <div class="insight-text">The RMS Titanic struck an iceberg at 11:40 PM and sank at 2:20 AM, with over 1,500 lives lost.</div>
        </div>
        <div class="insight-card">
            <span class="insight-icon">🚤</span>
            <div class="insight-title">20 Lifeboats</div>
            <div class="insight-text">The ship carried only 20 lifeboats — enough for 1,178 people, far fewer than the 2,224 aboard.</div>
        </div>
        <div class="insight-card">
            <span class="insight-icon">👩‍👧</span>
            <div class="insight-title">Women & Children First</div>
            <div class="insight-text">75% of women and 50% of children survived, compared to only about 20% of men.</div>
        </div>
        <div class="insight-card">
            <span class="insight-icon">👑</span>
            <div class="insight-title">Class Divide</div>
            <div class="insight-text">Nearly 63% of first-class passengers survived vs. only 25% of third-class passengers.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-bar">
    Built with <span class="gold">Machine Learning</span> &middot; Trained on the Titanic dataset &middot; For educational purposes only
</div>
""", unsafe_allow_html=True)