"""
AI-Powered Student Risk & Employability Predictor
Streamlit Interface — Group 2
Run with: streamlit run app.py
"""
 
import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
 
# ── Fix working directory ─────────────────────────────────────────────────────
BASE_DIR = r"c:\Users\jnyak\DS-FT15\Phase5\CAPSTONE---AI-POWERED-STUDENT-RISK-EMPLOYABILITY-PREDICTOR\Notebooks"
os.chdir(BASE_DIR)
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Risk & Employability Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');
 
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
 
    .main .block-container { background-color: #f7f8fc; }
 
    .hero {
        background: linear-gradient(135deg, #1a1f36 0%, #2d3561 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 { font-size: 2rem; font-weight: 600; margin: 0 0 0.5rem; }
    .hero p  { font-size: 1rem; opacity: 0.75; margin: 0; }
 
    .pred-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .pred-card h2 { font-size: 1.1rem; font-weight: 500; margin: 0 0 0.5rem; opacity: 0.7; }
    .pred-card h1 { font-size: 2rem; font-weight: 600; margin: 0; }
 
    .card-green  { background: #e8f8f0; color: #1a6b42; border: 1.5px solid #a8e6c5; }
    .card-red    { background: #fdecea; color: #7b1d1d; border: 1.5px solid #f5aca6; }
    .card-orange { background: #fff4e5; color: #7a3d00; border: 1.5px solid #ffcc80; }
    .card-blue   { background: #e8f0fe; color: #1a3a6b; border: 1.5px solid #a8c4f5; }
 
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e8eaf0;
    }
    .metric-box .val { font-size: 1.6rem; font-weight: 600; color: #2d3561; }
    .metric-box .lbl { font-size: 0.78rem; color: #888; margin-top: 2px; }
 
    .section-title {
        font-size: 1.15rem; font-weight: 600;
        color: #1a1f36; margin: 1.5rem 0 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e8eaf0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2d3561, #4a5db5);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 1rem;
        font-family: 'DM Sans', sans-serif;
        width: 100%; cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
 
    .info-box {
        background: #e8f0fe; border-radius: 10px;
        padding: 1rem 1.25rem; font-size: 0.88rem;
        color: #1a3a6b; margin-top: 1rem;
        border-left: 4px solid #4a5db5;
    }
    .warn-box {
        background: #fff4e5; border-radius: 10px;
        padding: 1rem 1.25rem; font-size: 0.88rem;
        color: #7a3d00; margin-top: 1rem;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)
 
# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_files = {
        'dropout':  os.path.join(BASE_DIR, 'models', 'dropout_best_model.pkl'),
        'academic': os.path.join(BASE_DIR, 'models', 'academic_best_model.pkl'),
        'employ':   os.path.join(BASE_DIR, 'models', 'employ_best_model.pkl'),
        'resume':   os.path.join(BASE_DIR, 'models', 'resume_best_model.pkl'),
    }
    loaded = {}
    for key, path in model_files.items():
        if os.path.exists(path):
            loaded[key] = joblib.load(path)
        else:
            loaded[key] = None
            st.warning(f"Model not found: {path}")
    return loaded
 
models = load_models()
 
def model_ready(key):
    return models.get(key) is not None
 
# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1> Student Risk & Employability Predictor</h1>
    <p>AI-powered early warning system for academic advisors and career counselors.</p>
</div>
""", unsafe_allow_html=True)
 
# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Navigation")
    page = st.radio(
        "",
        [" Home", " Dropout Risk", " Academic Risk",
         " Employability", " Resume Classifier"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Model status**")
    for label, key in [("Dropout model",       "dropout"),
                       ("Academic model",      "academic"),
                       ("Employability model", "employ"),
                       ("Resume NLP model",    "resume")]:
        icon = "🟢" if model_ready(key) else "🔴"
        st.markdown(f"{icon} {label}")
    st.divider()
    st.caption("Group 2 · Capstone Project")
 
# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == " Home":
    col1, col2, col3, col4 = st.columns(4)
    for col, label, val in zip(
        [col1, col2, col3, col4],
        ["Dropout dataset", "Employability dataset", "Resume dataset", "Models trained"],
        ["4,424", "1,164", "2,484", "4"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="val">{val}</div>
                <div class="lbl">{label}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown('<p class="section-title">How to use this tool</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ** Dropout Risk**
        Enter a student's enrolment demographics and first-semester performance.
        The model predicts whether the student will Graduate, remain Enrolled, or Drop out.
 
        ** Academic Risk**
        Enter GPA, attendance, and study habits.
        The model classifies the student as At Risk, Average, or High Performance.
        """)
    with col_b:
        st.markdown("""
        ** Employability**
        Enter skill scores across technical, soft-skill, and experiential dimensions.
        The model rates the student's employment readiness as Low, Medium, or High.
 
        ** Resume Classifier**
        Paste resume text and the NLP model predicts the most suitable job category
        from 25 industry sectors.
        """)
 
    st.markdown('<div class="info-box"> <strong>Tip:</strong> Use the sidebar to navigate between the four prediction modules.</div>', unsafe_allow_html=True)
 
# ═══════════════════════════════════════════════════════════════════════════════
# DROPOUT RISK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == " Dropout Risk":
    st.markdown('<p class="section-title"> Dropout Risk Predictor</p>', unsafe_allow_html=True)
    st.caption("Features: age at enrolment, admission grade, units approved, scholarship status, debtor status")
 
    col1, col2 = st.columns(2)
    with col1:
        age             = st.slider("Age at enrolment", 17, 60, 20)
        admission_grade = st.slider("Admission grade (0–200)", 0.0, 200.0, 120.0, step=0.5)
        total_units     = st.slider("Total units approved (Sem 1 + Sem 2)", 0, 30, 10)
    with col2:
        scholarship     = st.selectbox("Scholarship holder?", ["No (0)", "Yes (1)"])
        debtor          = st.selectbox("Has outstanding debt?", ["No (0)", "Yes (1)"])
        scholarship_val = 1 if "Yes" in scholarship else 0
        debtor_val      = 1 if "Yes" in debtor      else 0
 
    if st.button("Predict Dropout Risk", key="dropout_btn"):
        features = np.array([[age, admission_grade, total_units,
                               scholarship_val, debtor_val]])
        if model_ready("dropout"):
            model  = models["dropout"]
            pred   = model.predict(features)[0]
            labels = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
            result = labels.get(int(pred), str(pred))
            card   = {"Graduate": "card-green", "Dropout": "card-red",
                      "Enrolled": "card-orange"}.get(result, "card-blue")
 
            st.markdown(f"""
            <div class="pred-card {card}">
                <h2>Predicted outcome</h2>
                <h1>{result}</h1>
            </div>""", unsafe_allow_html=True)
 
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                st.markdown("**Prediction confidence**")
                for lbl, prob in zip(["Dropout", "Enrolled", "Graduate"], probs):
                    st.progress(float(prob), text=f"{lbl}: {prob*100:.1f}%")
 
            if result == "Dropout":
                st.markdown('<div class="warn-box"> <strong>High dropout risk detected.</strong> Consider referring this student to an academic advisor immediately.</div>', unsafe_allow_html=True)
            elif result == "Graduate":
                st.markdown('<div class="info-box"> <strong>Low dropout risk.</strong> Student is on track toward graduation.</div>', unsafe_allow_html=True)
        else:
            st.error("Dropout model not found in models/ folder.")
 
# ═══════════════════════════════════════════════════════════════════════════════
# ACADEMIC RISK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == " Academic Risk":
    st.markdown('<p class="section-title"> Academic Risk Predictor</p>', unsafe_allow_html=True)
    st.caption("Features: GPA, absences, hours studied, engagement score, study efficiency")
 
    col1, col2 = st.columns(2)
    with col1:
        gpa           = st.slider("GPA (0.0 – 4.0)", 0.0, 4.0, 2.5, step=0.01)
        absences      = st.slider("Number of absences", 0, 100, 10)
        hours_studied = st.slider("Weekly hours studied", 0, 60, 15)
    with col2:
        engagement = st.slider("Engagement score (0–100)", 0, 100, 60,
                               help="Composite of raised hands, resource visits, discussion activity")
        efficiency = st.slider("Study efficiency score (0–100)", 0, 100, 55,
                               help="Ratio of grades achieved to study hours invested")
 
    if st.button("Predict Academic Risk", key="academic_btn"):
        features = np.array([[gpa, absences, hours_studied, engagement, efficiency]])
        if model_ready("academic"):
            model  = models["academic"]
            pred   = model.predict(features)[0]
            labels = {0: " At Risk", 1: " Average", 2: " High Performance"}
            result = labels.get(int(pred), str(pred))
            card   = {0: "card-red", 1: "card-orange", 2: "card-green"}.get(int(pred), "card-blue")
 
            st.markdown(f"""
            <div class="pred-card {card}">
                <h2>Academic classification</h2>
                <h1>{result}</h1>
            </div>""", unsafe_allow_html=True)
 
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                st.markdown("**Prediction confidence**")
                for lbl, prob in zip(["At Risk", "Average", "High Performance"], probs):
                    st.progress(float(prob), text=f"{lbl}: {prob*100:.1f}%")
 
            if int(pred) == 0:
                st.markdown('<div class="warn-box"> <strong>Student flagged as At Risk.</strong> High absence count and low engagement are primary risk drivers. Recommend tutoring and attendance monitoring.</div>', unsafe_allow_html=True)
        else:
            st.error("Academic model not found in models/ folder.")

# ═══════════════════════════════════════════════════════════════════════════════
# EMPLOYABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == " Employability":
    st.markdown('<p class="section-title"> Employability Readiness Predictor</p>', unsafe_allow_html=True)
    st.caption("Features: GPA, technical score average, soft skills average, career readiness average")

    col1, col2 = st.columns(2)
    with col1:
        gpa           = st.slider("GPA (0.0 – 4.0)", 0.0, 4.0, 2.5, step=0.01)
        tech_score    = st.slider("Technical score average (0–10)",  0.0, 10.0, 5.0, step=0.1,
                                  help="Average of programming, domain knowledge, tool proficiency")
    with col2:
        soft_score    = st.slider("Soft skills average (0–10)", 0.0, 10.0, 5.0, step=0.1,
                                  help="Average of communication, teamwork, leadership, adaptability")
        career_score  = st.slider("Career readiness average (0–10)", 0.0, 10.0, 5.0, step=0.1,
                                  help="Average of internships, projects, certifications, market awareness")

    if st.button("Predict Employability", key="employ_btn"):
        features = np.array([[gpa, tech_score, soft_score, career_score]])
        if model_ready("employ"):
            model  = models["employ"]
            pred   = model.predict(features)[0]
            labels = {0: " Low Readiness", 1: " Medium Readiness", 2: " High Readiness"}
            result = labels.get(int(pred), str(pred))
            card   = {0: "card-red", 1: "card-orange", 2: "card-green"}.get(int(pred), "card-blue")

            st.markdown(f"""
            <div class="pred-card {card}">
                <h2>Employability classification</h2>
                <h1>{result}</h1>
            </div>""", unsafe_allow_html=True)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                st.markdown("**Prediction confidence**")
                for lbl, prob in zip(["Low", "Medium", "High"], probs):
                    st.progress(float(prob), text=f"{lbl} Readiness: {prob*100:.1f}%")

            if int(pred) == 0:
                st.markdown('<div class="warn-box"> <strong>Low employability readiness.</strong> Recommend technical workshops and internship placement before graduation.</div>', unsafe_allow_html=True)
            elif int(pred) == 2:
                st.markdown('<div class="info-box"> <strong>High employability readiness.</strong> Student is ready for job placement or graduate programme applications.</div>', unsafe_allow_html=True)
        else:
            st.error("Employability model not found in models/ folder.")
 
# ═══════════════════════════════════════════════════════════════════════════════
# RESUME CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == " Resume Classifier":
    st.markdown('<p class="section-title"> Resume Job Category Classifier</p>', unsafe_allow_html=True)
    st.caption("Adjust the topic scores below to classify the resume job category.")

    LABEL_MAP = {
        36: 'HR', 21: 'DESIGNER', 39: 'INFORMATION-TECHNOLOGY', 52: 'TEACHER',
        1: 'ADVOCATE', 12: 'BUSINESS-DEVELOPMENT', 35: 'HEALTHCARE', 32: 'FITNESS',
        2: 'AGRICULTURE', 11: 'BPO', 49: 'SALES', 18: 'CONSULTANT',
        22: 'DIGITAL-MEDIA', 5: 'AUTOMOBILE', 16: 'CHEF', 31: 'FINANCE',
        3: 'APPAREL', 28: 'ENGINEERING', 0: 'ACCOUNTANT', 17: 'CONSTRUCTION',
        47: 'PUBLIC-RELATIONS', 10: 'BANKING', 4: 'ARTS', 6: 'AVIATION',
        23: 'Data Science', 7: 'Advocate', 8: 'Arts', 54: 'Web Designing',
        42: 'Mechanical Engineer', 51: 'Sales', 38: 'Health and Fitness',
        19: 'Civil Engineer', 40: 'Java Developer', 15: 'Business Analyst',
        50: 'SAP Developer', 9: 'Automation Testing', 30: 'Electrical Engineering',
        45: 'Operations Manager', 48: 'Python Developer', 26: 'DevOps Engineer',
        44: 'Network Security Engineer', 46: 'PMO', 25: 'Database',
        37: 'Hadoop', 29: 'ETL Developer', 27: 'DotNet Developer',
        14: 'Blockchain', 53: 'Testing', 33: 'Frontend Developer',
        13: 'Backend Developer', 24: 'Data Scientist', 34: 'Full Stack Developer',
        43: 'Mobile App Developer (iOS/Android)', 41: 'Machine Learning Engineer',
        20: 'Cloud Engineer'
    }

    topic_values = []
    topic_labels = [
        "Technical Programming",
        "Business & Management",
        "Data & Analytics",
        "Design & Creativity",
        "Healthcare & Science",
        "Education & Training",
        "Finance & Accounting",
        "Operations & Logistics",
        "Sales & Marketing",
        "Infrastructure & Networks",
    ]

    st.markdown("**Topic Scores** — slide to reflect the strength of each theme in the resume")

    col1, col2 = st.columns([3, 1])
    for i, label in enumerate(topic_labels):
        with col1:
            val = st.slider(f"Topic {i} — {label}", -1.0, 1.0, 0.0,
                            step=0.01, key=f"topic_{i}")
        with col2:
            st.metric(label=f"Topic {i} value", value=f"{val:.2f}")
        topic_values.append(val)

    st.divider()
    has_exp     = st.selectbox("Has work experience?", ["No (0)", "Yes (1)"])
    has_exp_val = 1 if "Yes" in has_exp else 0

    if st.button("Classify Resume", key="resume_btn"):
        features = np.array([topic_values + [has_exp_val]])

        if model_ready("resume"):
            model      = models["resume"]
            pred_code  = int(model.predict(features)[0])
            pred_label = LABEL_MAP.get(pred_code, f"Category {pred_code}")

            st.markdown(f"""
            <div class="pred-card card-blue">
                <h2>Predicted job category</h2>
                <h1>{pred_label}</h1>
            </div>""", unsafe_allow_html=True)

            if hasattr(model, "predict_proba"):
                probs    = model.predict_proba(features)[0]
                top5_idx = np.argsort(probs)[::-1][:5]
                st.markdown("**Top 5 predicted categories**")
                for idx in top5_idx:
                    code  = int(model.classes_[idx])
                    label = LABEL_MAP.get(code, f"Category {code}")
                    st.progress(float(probs[idx]),
                                text=f"{label}: {probs[idx]*100:.1f}%")

            st.markdown('<div class="info-box"> Topic scores represent the strength of each theme in the resume. A score closer to 1.0 means strong presence of that topic; closer to -1.0 means weak or absent.</div>', unsafe_allow_html=True)
        else:
            st.error("Resume model not found in models/ folder.")