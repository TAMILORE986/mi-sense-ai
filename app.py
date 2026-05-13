import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
import os
import io
import json

# ── Configuration ──────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="MI-Sense AI | Advanced Cardiac Diagnostics",
    page_icon="🏥"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em;
                       background-color: #ff4b4b; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# ── XGBoost feature names — must match training data columns exactly ───────────
# These are the columns produced by simulate_tabular_data_full.py
XGB_FEATURE_NAMES = ['age', 'sex', 'hs_cTnT', 'systolic_bp', 'bmi', 'height', 'weight']


# ── Helper functions ───────────────────────────────────────────────────────────

def preprocess_ecg_signal(signal, sampling_rate=500, target_length=1000):
    """Bandpass filter + normalise + resize to target_length samples."""
    nyquist = 0.5 * sampling_rate
    low  = 0.5  / nyquist
    high = 40.0 / nyquist
    b, a = butter(3, [low, high], btype='band')
    filtered = filtfilt(b, a, signal, axis=0)
    mean, std = np.mean(filtered, axis=0), np.std(filtered, axis=0)
    norm = (filtered - mean) / (std + 1e-9)
    if norm.shape[0] > target_length:
        step = norm.shape[0] // target_length
        return norm[::step][:target_length]
    elif norm.shape[0] < target_length:
        return np.pad(norm, ((0, target_length - norm.shape[0]), (0, 0)), mode='constant')
    return norm


def load_ecg_from_upload(uploaded_file):
    """
    Load a 12-lead ECG from an uploaded file.
    Supports:
      • .npy  – numpy array of shape (samples, 12)
      • .csv  – comma/tab separated, 12 columns (one per lead)
    Returns a numpy array (samples, 12) or None on failure.
    """
    if uploaded_file is None:
        return None
    try:
        name = uploaded_file.name.lower()
        if name.endswith('.npy'):
            data = np.load(io.BytesIO(uploaded_file.read()))
        elif name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, header=None).values
        else:
            st.error("Unsupported file type. Please upload a .npy or .csv file.")
            return None

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.shape[1] != 12:
            st.warning(
                f"ECG has {data.shape[1]} column(s). Expected 12 leads. "
                "Padding/truncating to 12 columns."
            )
            if data.shape[1] < 12:
                data = np.pad(data, ((0, 0), (0, 12 - data.shape[1])), mode='constant')
            else:
                data = data[:, :12]

        return data.astype(np.float32)

    except Exception as e:
        st.error(f"Failed to load ECG file: {e}")
        return None


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cnn_lstm = None
    if os.path.exists('mi_detection_model_full.h5'):
        cnn_lstm = tf.keras.models.load_model('mi_detection_model_full.h5')

    xgboost_model = None
    if os.path.exists('xgboost_model.json'):
        xgboost_model = xgb.XGBClassifier()
        xgboost_model.load_model('xgboost_model.json')

    # MI type classifier (optional — loads if trained)
    mi_type_model = None
    if os.path.exists('mi_type_classifier.json'):
        mi_type_model = xgb.XGBClassifier()
        mi_type_model.load_model('mi_type_classifier.json')

    # Label map for MI types
    mi_type_labels = None
    if os.path.exists('mi_type_label_map.json'):
        with open('mi_type_label_map.json') as f:
            raw = json.load(f)
            mi_type_labels = {int(k): v for k, v in raw.items()}

    return cnn_lstm, xgboost_model, mi_type_model, mi_type_labels


cnn_lstm_model, xgb_model, mi_type_model, mi_type_labels = load_models()

# Normalisation ranges for CNN-LSTM clinical features
# ORDER must exactly match simulate_tabular_data_full.py column order: age, sex, hs_cTnT, systolic_bp, bmi, height, weight
CNN_SCALER = {
    'age':         {'min': 18,  'max': 90},
    'sex':         {'min': 0,   'max': 1},
    'hs_cTnT':     {'min': 2,   'max': 1000},
    'systolic_bp': {'min': 80,  'max': 200},
    'bmi':         {'min': 15,  'max': 50},
    'height':      {'min': 140, 'max': 210},
    'weight':      {'min': 40,  'max': 160},
}
CNN_FEATURES = list(CNN_SCALER.keys())  # 7 features — must match training data exactly


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
    st.title("MI-Sense AI")
    st.markdown("---")

    selected_model = st.selectbox(
        "🧠 Select AI Engine",
        ["CNN-LSTM (Multimodal)", "XGBoost (Clinical)"]
    )
    st.markdown("---")

    # ── ECG Upload ─────────────────────────────────────────────────────────────
    st.header("📤 Upload 12-Lead ECG")
    st.caption("Accepted formats: .npy (NumPy) or .csv (12 columns, one per lead)")
    ecg_upload = st.file_uploader(
        "Upload ECG signal file",
        type=["npy", "csv"],
        help="Array shape: (samples, 12). For CSV, 12 columns — one per lead, no header."
    )
    uploaded_ecg = load_ecg_from_upload(ecg_upload)
    if uploaded_ecg is not None:
        st.success(f"✅ ECG loaded: {uploaded_ecg.shape[0]} samples × {uploaded_ecg.shape[1]} leads")
    else:
        st.info("ℹ️ No ECG uploaded — synthetic signal will be used for CNN-LSTM mode.")

    st.markdown("---")

    # ── Patient Vitals ─────────────────────────────────────────────────────────
    st.header("📋 Patient Vitals")

    st.caption("🔬 Model inputs")
    age    = st.slider("Age (years)",               18,   90,  60)
    sex    = st.selectbox("Sex",                    ["Female", "Male"])
    height = st.slider("Height (cm)",              140,  210, 170)
    weight = st.slider("Weight (kg)",               40,  160,  70)
    bps    = st.slider("Systolic BP (mmHg)",        80,  200, 120)
    trop   = st.slider("Troponin T — hs-cTnT (ng/L)", 2, 1000, 14)

    bmi = weight / ((height / 100) ** 2)

    st.markdown("---")
    st.caption("📊 Additional clinical observations")
    bpd   = st.slider("Diastolic BP (mmHg)",        50,  120,  80)
    hr    = st.slider("Heart Rate (bpm)",            40,  150,  72)
    rr    = st.slider("Respiratory Rate (br/min)",   10,   30,  16)
    spo2  = st.slider("SpO2 (%)",                    85,  100,  98)
    temp  = st.slider("Temperature (°C)",          35.0, 40.0, 36.6)
    chol  = st.slider("Cholesterol (mg/dL)",        100,  300, 200)
    gluc  = st.slider("Glucose (mg/dL)",             70,  200, 100)

    # ── Build dataframes for each model ───────────────────────────────────────

    # XGBoost: must exactly match XGB_FEATURE_NAMES & training column order
    xgb_input = {
        'age':         age,
        'sex':         1 if sex == "Male" else 0,
        'hs_cTnT':     trop,
        'systolic_bp': bps,
        'bmi':         round(bmi, 2),
        'height':      height,
        'weight':      weight,
    }
    xgb_df = pd.DataFrame([xgb_input], columns=XGB_FEATURE_NAMES)

    # CNN-LSTM: 5 clinical sub-features, min-max scaled
    cnn_raw = {k: xgb_input[k] for k in CNN_FEATURES}
    cnn_df  = pd.DataFrame([cnn_raw])
    for col in CNN_FEATURES:
        mn = CNN_SCALER[col]['min']
        mx = CNN_SCALER[col]['max']
        cnn_df[col] = (cnn_df[col] - mn) / (mx - mn + 1e-9)
    cnn_scaled = cnn_df.values  # shape (1, 5)


# ── Main Dashboard ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🏥 Cardiac Diagnostic Dashboard")
    st.info("System Status: Online  |  Model Ready  |  Realistic Performance Mode")

    tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Results", "📈 ECG Visualisation", "🔍 AI Explainability"])

    # ── Tab 1 – Diagnostic Results ─────────────────────────────────────────────
    with tab1:
        if st.button("🚀 RUN FULL DIAGNOSTIC"):
            with st.spinner("Analysing multimodal data…"):

                if selected_model == "CNN-LSTM (Multimodal)":
                    if cnn_lstm_model is None:
                        st.error("CNN-LSTM model file not found (mi_detection_model_full.h5).")
                        st.stop()

                    # Use uploaded ECG if available, else generate synthetic
                    if uploaded_ecg is not None:
                        raw_ecg = uploaded_ecg
                    else:
                        raw_ecg = np.random.randn(1000, 12).astype(np.float32)
                        st.warning("No ECG uploaded — using synthetic signal for demonstration.")

                    processed_ecg = preprocess_ecg_signal(raw_ecg)
                    ecg_batch     = np.expand_dims(processed_ecg, axis=0)   # (1, 1000, 12)

                    prob = float(cnn_lstm_model.predict([ecg_batch, cnn_scaled])[0][0])

                else:  # XGBoost
                    if xgb_model is None:
                        st.error("XGBoost model file not found (xgboost_model.json).")
                        st.stop()
                    prob = float(xgb_model.predict_proba(xgb_df)[0][1])

                # ── Gauge chart ────────────────────────────────────────────────
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "MI Risk Probability (%)", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar':  {'color': "#ff4b4b" if prob > 0.5 else "#2ecc71"},
                        'steps': [
                            {'range': [0,  30], 'color': "#e8f8f5"},
                            {'range': [30, 70], 'color': "#fef9e7"},
                            {'range': [70, 100], 'color': "#fdedec"},
                        ],
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

                if prob > 0.5:
                    st.error(f"### ⚠️ HIGH RISK DETECTED ({prob:.1%})\n"
                             "Patient shows significant biomarkers for Myocardial Infarction. "
                             "Immediate cardiology consult recommended.")

                    # ── MI Type Classification ─────────────────────────────
                    st.markdown("---")
                    st.subheader("🫀 MI Type Classification")

                    if mi_type_model is not None and mi_type_labels is not None:
                        type_probs   = mi_type_model.predict_proba(xgb_df)[0]
                        type_pred    = int(np.argmax(type_probs))
                        type_name    = mi_type_labels[type_pred]
                        type_conf    = float(type_probs[type_pred])

                        # Colour-coded badge
                        badge_color = {
                            'Inferior MI':              '#e74c3c',
                            'Anterior MI':              '#e67e22',
                            'Anteroseptal MI':          '#d35400',
                            'Anterolateral MI':         '#c0392b',
                            'Inferolateral MI':         '#8e44ad',
                            'Inferoposterior MI':       '#2980b9',
                            'Inferoposterolateral MI':  '#1abc9c',
                            'Lateral MI':               '#16a085',
                            'Posterior MI':             '#27ae60',
                        }.get(type_name, '#7f8c8d')

                        st.markdown(f"""
                        <div style="
                            background:{badge_color}22;
                            border-left: 5px solid {badge_color};
                            padding: 16px 20px;
                            border-radius: 8px;
                            margin-bottom: 12px;
                        ">
                            <div style="font-size:13px; color:gray; margin-bottom:4px;">Predicted MI Type</div>
                            <div style="font-size:26px; font-weight:700; color:{badge_color};">{type_name}</div>
                            <div style="font-size:13px; color:gray; margin-top:6px;">
                                Confidence: {type_conf:.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability bar chart for all types
                        type_df = pd.DataFrame({
                            'MI Type':      [mi_type_labels[i] for i in range(len(type_probs))],
                            'Probability':  type_probs
                        }).sort_values('Probability', ascending=True)

                        fig_type = go.Figure(go.Bar(
                            x=type_df['Probability'] * 100,
                            y=type_df['MI Type'],
                            orientation='h',
                            marker_color=[
                                badge_color if t == type_name else '#bdc3c7'
                                for t in type_df['MI Type']
                            ],
                            text=[f"{p:.1f}%" for p in type_df['Probability'] * 100],
                            textposition='outside'
                        ))
                        fig_type.update_layout(
                            title="MI Type Probability Distribution",
                            xaxis_title="Probability (%)",
                            xaxis=dict(range=[0, 110]),
                            height=350,
                            margin=dict(l=10, r=10, t=40, b=10)
                        )
                        st.plotly_chart(fig_type, use_container_width=True)

                        # Clinical description
                        descriptions = {
                            'Inferior MI':              'Affects the inferior wall of the left ventricle. Associated with right coronary artery (RCA) occlusion. Changes seen in leads II, III, and aVF.',
                            'Anterior MI':              'Affects the anterior wall of the left ventricle. Associated with left anterior descending artery (LAD) occlusion. Changes seen in leads V1–V4.',
                            'Anteroseptal MI':          'Affects the anterior and septal walls. LAD occlusion proximal to septal branches. Changes in V1–V3.',
                            'Anterolateral MI':         'Affects the anterior and lateral walls. Large LAD or combined LAD/LCx territory. Changes in V1–V6, I, aVL.',
                            'Inferolateral MI':         'Affects inferior and lateral walls. Associated with RCA or LCx occlusion. Changes in II, III, aVF, V5–V6.',
                            'Inferoposterior MI':       'Affects inferior and posterior walls. RCA occlusion. Reciprocal changes in V1–V2.',
                            'Inferoposterolateral MI':  'Large territory involving inferior, posterior, and lateral walls. Dominant RCA or LCx territory.',
                            'Lateral MI':               'Affects lateral wall. Left circumflex artery (LCx) occlusion. Changes in I, aVL, V5–V6.',
                            'Posterior MI':             'Affects posterior wall. Often missed — presents with reciprocal changes (tall R, ST depression) in V1–V2.',
                        }
                        desc = descriptions.get(type_name, '')
                        if desc:
                            st.info(f"**About {type_name}:** {desc}")

                    else:
                        # Type model not trained yet — show rule-based hint
                        st.warning("MI type classifier not yet trained. Run `python train_mi_type_classifier.py` to enable this feature.")
                        st.markdown("""
                        **General MI Type Indicators (based on ECG leads):**
                        | ECG Leads Affected | Suspected MI Type |
                        |---|---|
                        | II, III, aVF | Inferior MI |
                        | V1 – V4 | Anterior MI |
                        | I, aVL, V5 – V6 | Lateral MI |
                        | V1 – V2 (reciprocal) | Posterior MI |
                        | V1 – V6, I, aVL | Anterolateral MI |
                        """)

                else:
                    st.success(f"### ✅ LOW RISK DETECTED ({prob:.1%})\n"
                               "Patient vitals and signals are within normal MI-risk variance.")

    # ── Tab 2 – ECG Visualisation ──────────────────────────────────────────────
    with tab2:
        st.subheader("12-Lead ECG Signal Analysis")

        if uploaded_ecg is not None:
            ecg_display = preprocess_ecg_signal(uploaded_ecg)
            title_suffix = "(Uploaded Patient ECG)"
        else:
            ecg_display  = np.random.randn(1000, 12).astype(np.float32)
            title_suffix = "(Synthetic Demo — upload an ECG for real analysis)"

        t = np.linspace(0, 10, ecg_display.shape[0])
        leads_to_show = ['Lead I', 'Lead II', 'Lead III', 'V1', 'V2', 'V3']
        n_leads = min(6, ecg_display.shape[1])

        fig_ecg, axes = plt.subplots(n_leads, 1, figsize=(12, 8), sharex=True)
        for i in range(n_leads):
            axes[i].plot(t, ecg_display[:, i], color='#c0392b', lw=0.8)
            axes[i].set_ylabel(leads_to_show[i], fontsize=9)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yticks([])
        axes[-1].set_xlabel("Time (s)")
        fig_ecg.suptitle(f"12-Lead ECG {title_suffix}", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig_ecg)

    # ── Tab 3 – SHAP Explainability ────────────────────────────────────────────
    with tab3:
        st.subheader("Feature Importance (SHAP)")
        st.write("This chart shows which clinical factors contributed most to the prediction.")

        if selected_model == "CNN-LSTM (Multimodal)":
            if cnn_lstm_model is None:
                st.error("CNN-LSTM model not loaded.")
            else:
                # Use synthetic ECG as fixed background for SHAP
                if uploaded_ecg is not None:
                    fixed_ecg = np.expand_dims(preprocess_ecg_signal(uploaded_ecg), 0)
                else:
                    fixed_ecg = np.expand_dims(preprocess_ecg_signal(np.random.randn(1000, 12)), 0)

                def cnn_predict_wrapper(clinical_array):
                    n = clinical_array.shape[0]
                    ecg_rep = np.repeat(fixed_ecg, n, axis=0)
                    return cnn_lstm_model.predict([ecg_rep, clinical_array], verbose=0)

                background = np.zeros((1, cnn_scaled.shape[1]))
                explainer  = shap.KernelExplainer(cnn_predict_wrapper, background)
                shap_vals  = explainer.shap_values(cnn_scaled, nsamples=50)

                vals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals

                fig_shap, _ = plt.subplots()
                shap.summary_plot(vals, cnn_scaled, feature_names=CNN_FEATURES,
                                  show=False, plot_type="bar")
                plt.title("SHAP Feature Importance (CNN-LSTM Clinical Branch)")
                plt.tight_layout()
                st.pyplot(fig_shap)

        else:  # XGBoost — FIX: use xgb_df (correct feature names & count)
            if xgb_model is None:
                st.error("XGBoost model not loaded.")
            else:
                explainer = shap.TreeExplainer(xgb_model)
                # xgb_df has same columns as training data — no column mismatch
                shap_vals = explainer.shap_values(xgb_df)

                fig_shap, _ = plt.subplots()
                shap.summary_plot(shap_vals, xgb_df, feature_names=XGB_FEATURE_NAMES,
                                  show=False, plot_type="bar")
                plt.title("SHAP Feature Importance (XGBoost Clinical Model)")
                plt.tight_layout()
                st.pyplot(fig_shap)


# ── Right column: Clinical Summary ────────────────────────────────────────────
with col2:
    st.subheader("📋 Clinical Summary")

    def stat_card(label, value):
        st.markdown(f"""
        <div style="
            padding: 12px 16px;
            margin-bottom: 10px;
            border-radius: 8px;
            border: 1px solid rgba(128,128,128,0.25);
            background: rgba(128,128,128,0.08);
        ">
            <div style="font-size:12px; color:gray; margin-bottom:4px;">{label}</div>
            <div style="font-size:20px; font-weight:600;">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("🔬 Model inputs")
    stat_card("Age",                f"{age} yrs")
    stat_card("Gender",             sex)
    stat_card("Height",             f"{height} cm")
    stat_card("Weight",             f"{weight} kg")
    stat_card("BMI",                f"{bmi:.1f} kg/m²")
    stat_card("Systolic BP",        f"{bps} mmHg")
    stat_card("Troponin T (hs-cTnT)", f"{trop} ng/L")

    st.markdown("<hr style='margin:10px 0; opacity:0.2'>", unsafe_allow_html=True)
    st.caption("📊 Additional observations")
    stat_card("Diastolic BP",       f"{bpd} mmHg")
    stat_card("Heart Rate",         f"{hr} bpm")
    stat_card("Respiratory Rate",   f"{rr} br/min")
    stat_card("SpO2",               f"{spo2}%")
    stat_card("Temperature",        f"{temp:.1f} °C")
    stat_card("Cholesterol",        f"{chol} mg/dL")
    stat_card("Glucose",            f"{gluc} mg/dL")

    st.markdown("---")
    ecg_status = "✅ Patient ECG uploaded" if uploaded_ecg is not None else "⚠️ No ECG uploaded (synthetic used)"
    st.info(ecg_status)

    st.info("""
**About this system**

Hybrid CNN-LSTM architecture processes real 12-lead ECG waveforms alongside clinical tabular data.
The XGBoost branch uses clinical biomarkers only.
Both models target realistic diagnostic accuracy (85–89%).
""")

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Normal_ECG.svg/1200px-Normal_ECG.svg.png",
        caption="Standard ECG Morphology Reference"
    )


st.markdown("---")
st.caption("MI-Sense AI | Final Year Project — Realistic Performance Version")
