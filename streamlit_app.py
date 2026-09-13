import pickle
from pathlib import Path

import numpy as np
import streamlit as st


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="RenalSense AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f3f7fb;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071d35, #104d76);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .brand {
        text-align: center;
        padding: 1rem 0 1.5rem;
    }

    .brand-icon {
        font-size: 3.5rem;
    }

    .brand-title {
        font-size: 1.6rem;
        font-weight: 800;
    }

    .brand-subtitle {
        font-size: 0.85rem;
        opacity: 0.75;
    }

    .header {
        background: linear-gradient(135deg, #092b4c, #1682b7);
        color: white;
        padding: 2.5rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(20, 70, 110, 0.18);
    }

    .header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
    }

    .header p {
        font-size: 1.05rem;
        margin-top: 0.7rem;
        opacity: 0.9;
    }

    .card {
        background: white;
        border: 1px solid #e1eaf2;
        border-radius: 18px;
        padding: 1.4rem;
        min-height: 145px;
        box-shadow: 0 7px 20px rgba(20, 60, 100, 0.06);
    }

    .card-icon {
        font-size: 1.8rem;
    }

    .card h3 {
        color: #102f4f;
        font-size: 1.05rem;
        margin: 0.5rem 0;
    }

    .card p {
        color: #718096;
        font-size: 0.9rem;
    }

    .section-title {
        color: #102f4f;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }

    .section-text {
        color: #718096;
        margin-bottom: 1rem;
    }

    [data-testid="stForm"] {
        background: white;
        padding: 1.8rem;
        border: 1px solid #e1eaf2;
        border-radius: 22px;
        box-shadow: 0 8px 24px rgba(20, 60, 100, 0.07);
    }

    label {
        color: #304b68 !important;
        font-weight: 600 !important;
    }

    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #092b4c, #1682b7);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 800;
        padding: 0.9rem;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #061d33, #10668e);
    }

    .result {
        padding: 2rem;
        border-radius: 22px;
        margin-top: 1rem;
        box-shadow: 0 8px 24px rgba(20, 60, 100, 0.08);
    }

    .result h2 {
        margin: 0 0 0.6rem;
        font-size: 1.8rem;
        font-weight: 800;
    }

    .result p {
        margin: 0;
        font-size: 1rem;
    }

    .positive {
        background: linear-gradient(135deg, #fff1f1, #ffe0e0);
        border: 1px solid #efb4b4;
        color: #991b1b;
    }

    .negative {
        background: linear-gradient(135deg, #effcf4, #dcf7e6);
        border: 1px solid #a8dfbd;
        color: #166534;
    }

    .info-box {
        background: white;
        border: 1px solid #e1eaf2;
        border-radius: 18px;
        padding: 1.5rem;
        min-height: 220px;
        box-shadow: 0 6px 18px rgba(20, 60, 100, 0.05);
    }

    .info-box h3 {
        color: #102f4f;
    }

    .info-box li {
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    .footer {
        text-align: center;
        color: #8291a3;
        font-size: 0.85rem;
        padding-top: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

SCALER_PATH = (
    BASE_DIR / "encoders_scalers" / "min-max-scaler.pickle"
)

MODEL_PATH = (
    BASE_DIR / "models_cache" / "decision tree max-depth=5.pickle"
)

COVER_IMAGE_PATH = BASE_DIR / "assets" / "kidney_cover.png"


# =========================================================
# MODEL LOADING
# =========================================================

def load_object(path):
    with open(path, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_resources():
    scaler = load_object(SCALER_PATH)
    model = load_object(MODEL_PATH)
    return scaler, model


# =========================================================
# PREDICTION
# =========================================================

def predict_target(gravity, ph, osmo, cond, urea, calc):
    scaler, model = load_resources()

    hydrogen_ion = 10 ** (-ph)
    acidic_alkaline = int(ph < 7.0)
    calc_osmo = calc * osmo
    cond_osmo_ratio = cond / osmo
    urea_gravity_ratio = urea / gravity

    features = np.array(
        [
            gravity,
            ph,
            osmo,
            cond,
            urea,
            calc,
            hydrogen_ion,
            acidic_alkaline,
            calc_osmo,
            cond_osmo_ratio,
            urea_gravity_ratio,
        ]
    ).reshape(1, -1)

    features[:, 0:-3] = scaler.transform(features[:, 0:-3])

    return model.predict(features)[0]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    if COVER_IMAGE_PATH.exists():
        st.image(str(COVER_IMAGE_PATH), use_container_width=True)

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🩺</div>
            <div class="brand-title">RenalSense AI</div>
            <div class="brand-subtitle">
                Intelligent Urine Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### 📊 Dashboard")
    st.info(
        "Enter laboratory urine-analysis values and generate "
        "a machine-learning prediction."
    )

    st.markdown("---")

    st.markdown("### 🤖 Model Information")
    st.write("**Algorithm:** Decision Tree")
    st.write("**Maximum depth:** 5")
    st.write("**Input values:** 6")
    st.write("**Project type:** Research prototype")

    st.markdown("---")

    st.markdown("### 🔐 Privacy")
    st.caption(
        "Do not enter names, phone numbers, or other personal information."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="header">
        <h1>Kidney Stone Risk Analysis</h1>
        <p>
            A modern machine-learning dashboard for analyzing urine
            parameters and estimating kidney stone risk.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SUMMARY CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="card">
            <div class="card-icon">🔬</div>
            <h3>Laboratory Data</h3>
            <p>Six urine-analysis measurements.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <div class="card-icon">🧠</div>
            <h3>AI Technology</h3>
            <p>Trained Decision Tree classifier.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <div class="card-icon">⚡</div>
            <h3>Fast Results</h3>
            <p>Generate predictions within seconds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class="card">
            <div class="card-icon">🛡️</div>
            <h3>Research Focus</h3>
            <p>Designed for educational use.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# MAIN TABS
# =========================================================

analysis_tab, about_tab = st.tabs(
    ["🧪 Perform Analysis", "📘 About the Project"]
)


# =========================================================
# ANALYSIS TAB
# =========================================================

with analysis_tab:

    st.markdown(
        '<div class="section-title">Enter Urine-Analysis Values</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-text">
            Enter the values from a urine laboratory report.
            All fields are required for prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("analysis_form"):

        st.markdown("#### 🧪 Primary Measurements")

        c1, c2, c3 = st.columns(3)

        with c1:
            gravity = st.number_input(
                "Specific Gravity",
                min_value=1.0,
                max_value=1.1,
                value=1.018,
                format="%.3f",
                help="Density of urine relative to water.",
            )

        with c2:
            ph = st.number_input(
                "Urine pH",
                min_value=0.0,
                max_value=14.0,
                value=6.02,
                format="%.2f",
                help="Acidity or alkalinity of urine.",
            )

        with c3:
            osmo = st.number_input(
                "Osmolarity",
                min_value=180.0,
                max_value=1300.0,
                value=612.84,
                format="%.2f",
                help="Concentration of dissolved particles.",
            )

        st.markdown("#### ⚗️ Additional Measurements")

        c4, c5, c6 = st.columns(3)

        with c4:
            cond = st.number_input(
                "Conductivity",
                min_value=5.0,
                max_value=40.0,
                value=20.8,
                format="%.2f",
                help="Electrical conductivity of urine.",
            )

        with c5:
            urea = st.number_input(
                "Urea Concentration",
                min_value=10.0,
                max_value=620.0,
                value=226.4,
                format="%.2f",
                help="Urea concentration in urine.",
            )

        with c6:
            calc = st.number_input(
                "Calcium Concentration",
                min_value=0.1,
                max_value=15.0,
                value=4.13,
                format="%.2f",
                help="Calcium concentration in urine.",
            )

        submitted = st.form_submit_button(
            "🔍 Generate Kidney Stone Prediction",
            use_container_width=True,
        )

    if submitted:

        try:
            prediction = predict_target(
                gravity,
                ph,
                osmo,
                cond,
                urea,
                calc,
            )

            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True,
            )

            if prediction < 0.5:
                st.markdown(
                    """
                    <div class="result negative">
                        <h2>✅ Negative Prediction</h2>
                        <p>
                            The model did not identify a kidney-stone
                            pattern in the entered values.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="result positive">
                        <h2>⚠️ Positive Prediction</h2>
                        <p>
                            The model identified a pattern that may be
                            associated with kidney stone presence.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.info(
                "This is a machine-learning prediction and not a medical "
                "diagnosis. Consult a qualified healthcare professional "
                "for clinical interpretation."
            )

        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")


# =========================================================
# ABOUT TAB
# =========================================================

with about_tab:

    st.markdown(
        '<div class="section-title">About RenalSense AI</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "RenalSense AI is an educational machine-learning application "
        "that analyzes urine measurements and estimates the likelihood "
        "of kidney stone presence."
    )

    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            """
            <div class="info-box">
                <h3>🔬 Data Processing</h3>
                <ul>
                    <li>Urine data collection</li>
                    <li>Data cleaning</li>
                    <li>Feature engineering</li>
                    <li>Numerical scaling</li>
                    <li>Model prediction</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            """
            <div class="info-box">
                <h3>🤖 How It Works</h3>
                <ul>
                    <li>User enters laboratory values</li>
                    <li>Features are transformed</li>
                    <li>Trained model is loaded</li>
                    <li>Prediction is generated</li>
                    <li>Result is displayed</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(
        "Educational and research use only. This application does not "
        "replace laboratory testing, medical advice, or clinical diagnosis."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        RenalSense AI • Kidney Stone Prediction • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)