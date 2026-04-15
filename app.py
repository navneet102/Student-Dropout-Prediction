"""
app.py  —  Student Dropout Prediction · Streamlit Deployment
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────── Page config ────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0ff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(12px);
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(99,102,241,0.3);
    }
    .metric-card h3 {
        margin: 0 0 0.3rem 0;
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #a5b4fc;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
    }

    /* Result card */
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        animation: fadeIn 0.5s ease;
    }
    .dropout-card {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.3));
        border: 2px solid rgba(239,68,68,0.5);
        box-shadow: 0 0 30px rgba(239,68,68,0.2);
    }
    .safe-card {
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.3));
        border: 2px solid rgba(16,185,129,0.5);
        box-shadow: 0 0 30px rgba(16,185,129,0.2);
    }
    .result-card h2 {
        font-size: 2rem;
        margin: 0.5rem 0;
        color: #fff;
    }
    .result-card p {
        font-size: 1rem;
        opacity: 0.85;
        color: #e0e0ff;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #a5b4fc;
        font-size: 1.1rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #a5b4fc;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(165,180,252,0.2);
        padding-bottom: 0.4rem;
    }

    /* Streamlit buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 20px rgba(99,102,241,0.35);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        box-shadow: 0 6px 28px rgba(99,102,241,0.55);
        transform: translateY(-2px);
    }

    /* Sliders & inputs */
    .stSlider > div > div > div > div {
        background: #6366f1 !important;
    }
    label {
        color: #c7d2fe !important;
        font-size: 0.88rem !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Probability bar */
    .prob-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 99px;
        height: 14px;
        margin: 0.5rem 0 1rem;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.6s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── Load model ─────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

payload = load_model()
model       = payload["model"]
model_name  = payload["model_name"]
scaler      = payload["scaler"]
pca         = payload["pca"]
pca_cols    = payload["pca_cols"]
feature_names = payload["feature_names"]
accuracy    = payload["accuracy"]
all_models  = payload.get("all_models", {})

# ─────────────────────────── Hero header ────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Dropout Predictor</h1>
        <p>AI-powered early warning system using supervised machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── Sidebar — model info ────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Info")
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>Active Model</h3>
            <div class="value">{model_name}</div>
        </div>
        <div class="metric-card">
            <h3>Test Accuracy</h3>
            <div class="value">{accuracy * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if all_models:
        st.markdown("### All Models")
        rows = []
        for name, info in all_models.items():
            rows.append({"Model": name, "Accuracy": f"{info['accuracy'] * 100:.1f}%"})
        df_models = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df_models, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280'>Dataset: 4,424 students · 17 features after preprocessing</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────── Main layout ─────────────────────────────────────
col_form, col_result = st.columns([1.1, 0.9], gap="large")

# ── Left column: input form ──────────────────────────────────────────────────
with col_form:
    st.markdown("<div class='section-header'>Personal Information</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        marital_status = st.selectbox(
            "Marital Status",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "Single", 2: "Married", 3: "Widower",
                4: "Divorced", 5: "Facto Union", 6: "Legally Separated"
            }[x],
            index=0,
        )
        gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        age = st.slider("Age at Enrollment", min_value=17, max_value=70, value=20)
    with c2:
        international = st.selectbox("International Student", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        displaced = st.selectbox("Displaced", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        educational_special_needs = st.selectbox("Educational Special Needs", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    st.markdown("<div class='section-header'>Academic Background</div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        application_mode = st.slider("Application Mode", 1, 18, 8)
        application_order = st.slider("Application Order", 0, 9, 1)
        previous_qualification = st.slider("Previous Qualification", 1, 17, 1)
    with c4:
        course = st.selectbox(
            "Course",
            options=list(range(1, 18)),
            format_func=lambda x: {
                1: "Biofuel Production Technologies",
                2: "Animation & Multimedia Design",
                3: "Social Service (evening)",
                4: "Agronomy",
                5: "Communication Design",
                6: "Veterinary Nursing",
                7: "Informatics Engineering",
                8: "Equiniculture",
                9: "Management",
                10: "Social Service",
                11: "Tourism",
                12: "Nursing",
                13: "Oral Hygiene",
                14: "Advertising & Marketing Mgmt",
                15: "Journalism & Communication",
                16: "Basic Education",
                17: "Management (evening)",
            }[x],
        )
        daytime_attendance = st.selectbox("Attendance", options=[1, 0], format_func=lambda x: "Daytime" if x == 1 else "Evening")
        mothers_qualification = st.slider("Mother's Qualification", 1, 29, 13)
        fathers_occupation = st.slider("Father's Occupation", 1, 46, 10)

    st.markdown("<div class='section-header'>Financial & Scholarship</div>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        debtor = st.selectbox("Debtor", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        tuition_fees = st.selectbox("Tuition Fees Up To Date", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with c6:
        scholarship = st.selectbox("Scholarship Holder", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    st.markdown("<div class='section-header'>Curricular Units (1st & 2nd Semester)</div>", unsafe_allow_html=True)
    st.caption("Enter average academic performance across both semesters")
    c7, c8 = st.columns(2)
    pca_inputs = {}
    pca_labels = {
        "Curricular units 1st sem (credited)": ("1st Sem Credited", 0, 20, 2),
        "Curricular units 1st sem (enrolled)": ("1st Sem Enrolled", 0, 26, 6),
        "Curricular units 1st sem (evaluations)": ("1st Sem Evaluations", 0, 45, 8),
        "Curricular units 1st sem (without evaluations)": ("1st Sem (no eval)", 0, 12, 0),
        "Curricular units 1st sem (approved)": ("1st Sem Approved", 0, 26, 5),
        "Curricular units 1st sem (grade)": ("1st Sem Grade", 0.0, 18.876, 12.0),
        "Curricular units 2nd sem (credited)": ("2nd Sem Credited", 0, 19, 0),
        "Curricular units 2nd sem (enrolled)": ("2nd Sem Enrolled", 0, 23, 6),
        "Curricular units 2nd sem (evaluations)": ("2nd Sem Evaluations", 0, 33, 8),
        "Curricular units 2nd sem (without evaluations)": ("2nd Sem (no eval)", 0, 12, 0),
        "Curricular units 2nd sem (approved)": ("2nd Sem Approved", 0, 20, 5),
        "Curricular units 2nd sem (grade)": ("2nd Sem Grade", 0.0, 18.572, 12.0),
    }
    pca_col_list = list(pca_labels.keys())
    for i, col_name in enumerate(pca_col_list):
        label, mn, mx, def_val = pca_labels[col_name]
        target_col = c7 if i % 2 == 0 else c8
        with target_col:
            if isinstance(def_val, float):
                pca_inputs[col_name] = st.number_input(label, min_value=float(mn), max_value=float(mx), value=float(def_val), step=0.1, format="%.2f")
            else:
                pca_inputs[col_name] = st.number_input(label, min_value=int(mn), max_value=int(mx), value=int(def_val))

# ── Right column: predict + result ──────────────────────────────────────────
with col_result:
    st.markdown("<div class='section-header'>Prediction</div>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Dropout Risk", use_container_width=True)

    if predict_btn:
        # Build PCA array
        pca_arr = np.array([[pca_inputs[c] for c in pca_cols]])
        pca_value = pca.transform(pca_arr)[0][0]

        # Build full feature row
        raw_features = {
            "Marital status": marital_status,
            "Application mode": application_mode,
            "Application order": application_order,
            "Course": course,
            "Daytime/evening attendance": daytime_attendance,
            "Previous qualification": previous_qualification,
            "Mother's qualification": mothers_qualification,
            "Father's occupation": fathers_occupation,
            "Displaced": displaced,
            "Educational special needs": educational_special_needs,
            "Debtor": debtor,
            "Tuition fees up to date": tuition_fees,
            "Gender": gender,
            "Scholarship holder": scholarship,
            "Age at enrollment": age,
            "International": international,
            "Curricular 1st and 2nd sem PCA": pca_value,
        }

        input_df = pd.DataFrame([raw_features])[feature_names]
        X_input = scaler.transform(input_df)
        prediction = model.predict(X_input)[0]

        # Probability (if available)
        prob_dropout = None
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_input)[0]
            prob_dropout = probas[1] if len(probas) > 1 else probas[0]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X_input)[0]
            prob_dropout = 1 / (1 + np.exp(-score))  # sigmoid approximation

        is_dropout = bool(prediction == 1)

        if is_dropout:
            emoji = "⚠️"
            label_text = "HIGH DROPOUT RISK"
            desc = "This student shows indicators associated with dropping out. Early intervention is recommended."
            card_class = "dropout-card"
            bar_color = "#ef4444"
        else:
            emoji = "✅"
            label_text = "LOW DROPOUT RISK"
            desc = "This student appears on track to graduate or remain enrolled."
            card_class = "safe-card"
            bar_color = "#10b981"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div style="font-size:3rem">{emoji}</div>
                <h2>{label_text}</h2>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if prob_dropout is not None:
            pct = prob_dropout * 100
            st.markdown(
                f"""
                <p style="color:#a5b4fc;font-size:0.9rem;margin-bottom:0.2rem">
                    Dropout Probability: <strong style="color:#fff">{pct:.1f}%</strong>
                </p>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{bar_color};"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Feature summary card
        st.markdown("<div class='section-header'>Feature Summary</div>", unsafe_allow_html=True)
        summary_data = {
            "Feature": list(raw_features.keys()),
            "Value": [f"{v:.2f}" if isinstance(v, float) else str(v) for v in raw_features.values()],
        }
        st.dataframe(pd.DataFrame(summary_data).set_index("Feature"), use_container_width=True, height=400)

    else:
        # Placeholder when no prediction yet
        st.markdown(
            """
            <div style="
                margin-top: 3rem;
                text-align: center;
                padding: 3rem 2rem;
                background: rgba(255,255,255,0.04);
                border-radius: 20px;
                border: 1px dashed rgba(165,180,252,0.3);
            ">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🎓</div>
                <p style="color:#a5b4fc; font-size:1.1rem; margin:0">
                    Fill in the student details on the left<br>and click <strong style="color:#fff">Predict</strong> to see the risk assessment.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────── Footer ─────────────────────────────────────────
st.markdown(
    """
    <hr style="border-color:rgba(255,255,255,0.08); margin-top:3rem"/>
    <p style="text-align:center; color:#4b5563; font-size:0.8rem">
    Student Dropout Prediction · Supervised Machine Learning · Dataset: 4,424 students
    </p>
    """,
    unsafe_allow_html=True,
)
