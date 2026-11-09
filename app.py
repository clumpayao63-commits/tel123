from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_model.pkl"
METRICS_PATH = ROOT / "model_metrics.json"

FEATURES = ["days_open", "current_workload", "submitted_hour",
            "request_type", "priority", "department"]
REQUIRED_COLUMNS = FEATURES + ["sla_breach"]
NUMERIC = ["days_open", "current_workload", "submitted_hour"]
CATEGORICAL = ["request_type", "priority", "department"]

st.set_page_config(page_title="Student Service Request Monitoring", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())

def validate_dataset(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    if df.empty:
        return False, "The uploaded CSV has no rows."
    return True, "Dataset schema is valid."

model = load_model()
metrics = load_metrics()

st.title("🎓 Student Service Request Monitoring")
st.caption("Decision-support prototype for identifying student service requests that may breach their service-level agreement (SLA).")
st.warning("Demonstration project: the included CSV and trained model use synthetic sample data. Replace them with your authorized midterm dataset, exact target/features, and original fitted pipeline before submission or real use.")

page = st.sidebar.selectbox("Navigate", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])

if page == "Home":
    st.subheader("System overview")
    st.write("This application supports student-services staff by organizing request data, exploring workload patterns, estimating possible SLA breaches, and prioritizing human review.")
    a, b, c = st.columns(3)
    a.metric("Prediction target", "SLA breach")
    b.metric("Model", metrics.get("model_name", "Not available"))
    c.metric("Model status", "Loaded" if model is not None else "Missing")
    st.markdown("**Workflow:** Upload and validate CSV → Explore request patterns → Generate risk estimates → Staff reviews the evidence → Authorized staff records the final decision.")
    st.info("A model prediction is not an automatic decision to reject, deny, or delay a student service request.")

elif page == "Dataset":
    st.subheader("Dataset upload and validation")
    uploaded = st.file_uploader("Upload a student service request CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            valid, message = validate_dataset(df)
            (st.success if valid else st.error)(message)
            st.dataframe(df.head(30), use_container_width=True)
            st.write(f"Rows: **{len(df)}** | Columns: **{len(df.columns)}**")
            st.write("Missing values by column")
            st.dataframe(df.isna().sum().rename("missing_count"))
            st.write(f"Duplicate rows: **{int(df.duplicated().sum())}**")
            if valid:
                st.session_state["uploaded_df"] = df
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    else:
        st.caption("Upload a CSV to inspect its schema, preview rows, missing values, and duplicates.")
        sample = pd.read_csv(ROOT / "student_data.csv")
        st.dataframe(sample.head(10), use_container_width=True)

elif page == "Data Analysis":
    st.subheader("Exploratory data analysis")
    uploaded_df = st.session_state.get("uploaded_df")
    df = uploaded_df if uploaded_df is not None else pd.read_csv(ROOT / "student_data.csv")
    valid, message = validate_dataset(df)
    if not valid:
        st.error(message)
    else:
        st.write("Descriptive statistics")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        st.write("SLA breach distribution")
        st.plotly_chart(px.histogram(df, x="sla_breach", color="sla_breach",
                         title="Requests by SLA breach class"), use_container_width=True)
        st.plotly_chart(px.box(df, x="priority", y="days_open", color="priority",
                         title="Days open by priority"), use_container_width=True)
        st.plotly_chart(px.scatter(df, x="current_workload", y="days_open",
                         color=df["sla_breach"].astype(str), hover_data=["request_type", "department"],
                         labels={"color": "SLA breach"}, title="Workload and request age"), use_container_width=True)

elif page == "Prediction":
    st.subheader("Predictive modeling")
    if model is None:
        st.error("student_model.pkl was not found. Run: python train_model.py")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                request_type = st.selectbox("Request type", ["Enrollment", "Transcript", "ID Card", "Scholarship", "Clearance", "Other"])
                priority = st.selectbox("Priority", ["Low", "Medium", "High"])
                department = st.selectbox("Department", ["Registrar", "Student Affairs", "Finance", "IT", "Guidance"])
            with col2:
                days_open = st.number_input("Days request has been open", min_value=0, max_value=365, value=5)
                current_workload = st.number_input("Current staff workload", min_value=0, max_value=200, value=25)
                submitted_hour = st.number_input("Submission hour (0–23)", min_value=0, max_value=23, value=10)
            submitted = st.form_submit_button("Estimate SLA breach risk")
        if submitted:
            row = pd.DataFrame([{
                "days_open": days_open, "current_workload": current_workload,
                "submitted_hour": submitted_hour, "request_type": request_type,
                "priority": priority, "department": department
            }], columns=FEATURES)
            prediction = int(model.predict(row)[0])
            st.metric("Estimated outcome", "Possible SLA breach" if prediction == 1 else "No breach predicted")
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(row)[0][1])
                st.metric("Estimated breach probability", f"{probability:.1%}")
                st.progress(probability)
            st.info("Recommended human follow-up: verify the request record, current queue, SLA policy, and any exceptional circumstances before taking action.")

elif page == "Dashboard":
    st.subheader("Decision-support dashboard")
    df = st.session_state.get("uploaded_df")
    if df is None:
        df = pd.read_csv(ROOT / "student_data.csv")
    valid, message = validate_dataset(df)
    if not valid:
        st.error(message)
    else:
        total = len(df)
        breaches = int(df["sla_breach"].sum())
        rate = breaches / total if total else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Total requests", total)
        c2.metric("Observed SLA breaches", breaches)
        c3.metric("Observed breach rate", f"{rate:.1%}")
        st.write("Review queue (highest age/workload first)")
        queue = df.copy()
        queue["review_priority_score"] = queue["days_open"] + 0.25 * queue["current_workload"]
        st.dataframe(queue.sort_values("review_priority_score", ascending=False).head(20), use_container_width=True)
        st.caption("The queue score is a simple operational sorting aid, not a model prediction or an authorization decision.")
        if metrics:
            st.write("Model evaluation metrics (synthetic sample training)")
            st.json({k: metrics.get(k) for k in ["accuracy", "precision", "recall", "f1_score"]})

elif page == "System Design":
    st.subheader("SAD integration")
    st.markdown("**Actors**")
    st.write("- Student: submits a service request and checks its status.\n- Student Services Staff: reviews request details and model risk indicators.\n- System Administrator: manages authorized access and system configuration.")
    st.markdown("**Functional requirements**")
    st.write("1. The system shall accept CSV uploads and validate required columns.\n2. The system shall display dataset previews and data-quality summaries.\n3. The system shall present descriptive statistics and interactive charts.\n4. The system shall estimate SLA-breach risk using the saved model pipeline.\n5. The system shall display a review queue to support staff follow-up.")
    st.markdown("**Non-functional requirements**")
    st.write("- Security: only authorized users should access identifiable student records.\n- Reliability: invalid files and missing model artifacts should produce clear errors.\n- Usability: charts, labels, and recommendations should be understandable to staff.")
    st.markdown("**Decision workflow**")
    st.code("Student submits request\\n        ↓\\nSystem stores request data\\n        ↓\\nSaved model estimates SLA-breach risk\\n        ↓\\nStaff reviews request and supporting details\\n        ↓\\nAuthorized staff makes and records final decision", language="text")
    st.caption("Access control and decision-record persistence must be implemented before production deployment.")
