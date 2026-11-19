import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import httpx
import joblib

from src.utils.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    METADATA_PATH,
    COMPARISON_METRICS_PATH,
    TEST_METRICS_PATH,
    FIGURES_DIR,
    BOOKING_CHANNELS,
    REMINDER_PREFERENCES,
    APPOINTMENT_TYPES,
    WEEKDAYS,
    MONTHS,
    DEFAULT_THRESHOLD,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
)
from src.models.explainability import explain_single_prediction
from src.models.evaluate import sweep_thresholds

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Hospital Appointment Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 14px;
        color: #4a5568;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #2b6cb0;
    }
    .metric-label {
        font-size: 12px;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-badge-low {
        background-color: #c6f6d5;
        color: #22543d;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-moderate {
        background-color: #feebc8;
        color: #7b341e;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #fed7d7;
        color: #742a2a;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    return pd.DataFrame()


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_test_metrics():
    if TEST_METRICS_PATH.exists():
        with open(TEST_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_comparison_metrics():
    if COMPARISON_METRICS_PATH.exists():
        return pd.read_csv(COMPARISON_METRICS_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_local_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def call_api_predict(payload: dict) -> dict:
    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.post(f"{API_BASE_URL}/predict", json=payload)
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass

    model = load_local_model()
    if model is None:
        raise RuntimeError("Neither API nor local model artifact is accessible.")

    input_df = pd.DataFrame([payload])
    threshold = payload.get("threshold", DEFAULT_THRESHOLD)
    prob = float(model.predict_proba(input_df)[0, 1])
    predicted_no_show = bool(prob >= threshold)

    if prob < RISK_THRESHOLD_LOW:
        risk_level = "Low"
    elif prob < RISK_THRESHOLD_HIGH:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    df_data = load_data()
    sample = df_data.head(200) if not df_data.empty else pd.DataFrame()
    explanation = {}
    if not sample.empty:
        try:
            explanation = explain_single_prediction(model, input_df, sample)
        except Exception:
            explanation = {}

    meta = load_metadata()
    return {
        "no_show_probability": round(prob, 4),
        "predicted_no_show": predicted_no_show,
        "risk_level": risk_level,
        "threshold": round(threshold, 4),
        "model_name": meta.get("model_name", "Local Trained Model"),
        "key_factors": explanation,
        "disclaimer": "Decision-support estimate requiring human review. Not a clinical diagnosis.",
    }


def render_page_overview(df: pd.DataFrame, meta: dict, test_m: dict):
    st.markdown('<div class="main-header">Hospital Appointment Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive Operational Overview and Real-Time Appointment Dynamics</div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("Processed dataset not found. Please run the training pipeline first.")
        return

    total_records = len(df)
    total_attended = int((df["no_show"] == 0).sum())
    total_missed = int((df["no_show"] == 1).sum())
    noshow_rate = total_missed / total_records if total_records > 0 else 0.0

    selected_model_name = meta.get("model_name", "Random Forest")
    test_f1 = test_m.get("metrics_at_default_threshold", {}).get("f1", 0.0)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Appointments</div><div class="metric-value">{total_records:,}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Attended</div><div class="metric-value">{total_attended:,}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Missed (No-Show)</div><div class="metric-value">{total_missed:,}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">No-Show Rate</div><div class="metric-value">{noshow_rate*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Active Model F1</div><div class="metric-value">{test_f1:.3f}</div><div style="font-size:11px;color:#718096;">{selected_model_name}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        status_df = pd.DataFrame({
            "Status": ["Attended", "No-Show (Missed)"],
            "Count": [total_attended, total_missed],
        })
        fig_donut = px.pie(
            status_df,
            names="Status",
            values="Count",
            hole=0.55,
            color="Status",
            color_discrete_map={"Attended": "#2b6cb0", "No-Show (Missed)": "#dd6b20"},
            title="Overall Attendance Proportion",
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        fig_donut.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        df_time = df.copy()
        df_time["appointment_month_num"] = pd.to_datetime(df_time["appointment_date"]).dt.month
        monthly_trend = (
            df_time.groupby(["appointment_month_num", "appointment_month"])["no_show"]
            .mean()
            .reset_index()
            .sort_values(by="appointment_month_num")
        )
        fig_trend = px.line(
            monthly_trend,
            x="appointment_month",
            y="no_show",
            markers=True,
            title="Monthly No-Show Rate Trend",
            labels={"appointment_month": "Month", "no_show": "No-Show Rate"},
        )
        fig_trend.update_yaxes(tickformat=".1%")
        fig_trend.update_traces(line_color="#2b6cb0", line_width=3)
        fig_trend.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        weekday_order = [w for w in WEEKDAYS if w in df["appointment_weekday"].unique()]
        weekday_df = (
            df.groupby("appointment_weekday")["no_show"]
            .mean()
            .reindex(weekday_order)
            .reset_index()
        )
        fig_wk = px.bar(
            weekday_df,
            x="appointment_weekday",
            y="no_show",
            title="No-Show Rate by Weekday",
            labels={"appointment_weekday": "Weekday", "no_show": "Rate"},
            color="no_show",
            color_continuous_scale="Blues",
        )
        fig_wk.update_yaxes(tickformat=".1%")
        fig_wk.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_wk, use_container_width=True)

    with c2:
        hour_df = (
            df.groupby("appointment_hour")["no_show"]
            .mean()
            .reset_index()
            .sort_values(by="appointment_hour")
        )
        fig_hr = px.bar(
            hour_df,
            x="appointment_hour",
            y="no_show",
            title="No-Show Rate by Appointment Hour",
            labels={"appointment_hour": "Hour (24h)", "no_show": "Rate"},
            color="no_show",
            color_continuous_scale="Teal",
        )
        fig_hr.update_yaxes(tickformat=".1%")
        fig_hr.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_hr, use_container_width=True)

    with c3:
        bins = [-1, 0, 3, 7, 14, 30, 100]
        labels = ["Same Day (0)", "1-3 Days", "4-7 Days", "8-14 Days", "15-30 Days", "30+ Days"]
        df_lt = df.copy()
        df_lt["lead_time_cat"] = pd.cut(df_lt["lead_time_days"], bins=bins, labels=labels)
        lead_df = (
            df_lt.groupby("lead_time_cat", observed=False)["no_show"]
            .mean()
            .reset_index()
        )
        fig_lt = px.bar(
            lead_df,
            x="lead_time_cat",
            y="no_show",
            title="No-Show Rate by Lead Time Group",
            labels={"lead_time_cat": "Lead Time", "no_show": "Rate"},
            color="no_show",
            color_continuous_scale="Viridis",
        )
        fig_lt.update_yaxes(tickformat=".1%")
        fig_lt.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_lt, use_container_width=True)


def render_page_predict():
    st.markdown('<div class="main-header">Appointment Risk Assessment & Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Decision Support for Scheduling Staff and Administrative Teams</div>', unsafe_allow_html=True)

    tab_single, tab_batch = st.tabs(["Single Appointment Inference", "Batch File Evaluation"])

    with tab_single:
        with st.form("single_predict_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                lead_time_days = st.slider("Lead Time (Days between Booking & Appointment)", 0, 60, 10)
                appointment_weekday = st.selectbox("Appointment Weekday", WEEKDAYS, index=0)
                appointment_month = st.selectbox("Appointment Month", MONTHS, index=2)
                is_weekend = 1 if appointment_weekday in ["Saturday", "Sunday"] else 0

            with col2:
                appointment_hour = st.slider("Appointment Hour (24-Hour Clock)", 8, 17, 10)
                booking_channel = st.selectbox("Booking Channel", BOOKING_CHANNELS, index=0)
                reminder_preference = st.selectbox("Reminder Preference", REMINDER_PREFERENCES, index=0)

            with col3:
                appointment_type = st.selectbox("Appointment Type", APPOINTMENT_TYPES, index=0)
                previous_appointments = st.number_input("Previous Completed Appointments", min_value=0, max_value=50, value=3)
                previous_no_shows = st.number_input("Prior Missed Appointments", min_value=0, max_value=previous_appointments, value=0)

            st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
            thresh_col, btn_col = st.columns([2, 1])
            with thresh_col:
                threshold = st.slider(
                    "Classification Decision Threshold",
                    min_value=0.10,
                    max_value=0.90,
                    value=0.50,
                    step=0.05,
                    help="Appointments with predicted probability exceeding this threshold will be flagged as high-risk no-shows.",
                )
            with btn_col:
                st.write("")
                st.write("")
                submitted = st.form_submit_button("Calculate Risk Score", use_container_width=True)

        if submitted:
            if previous_no_shows > previous_appointments:
                st.error("Prior missed appointments cannot exceed previous completed appointments.")
                return

            payload = {
                "lead_time_days": int(lead_time_days),
                "appointment_weekday": str(appointment_weekday),
                "appointment_month": str(appointment_month),
                "appointment_hour": int(appointment_hour),
                "booking_channel": str(booking_channel),
                "reminder_preference": str(reminder_preference),
                "previous_appointments": int(previous_appointments),
                "previous_no_shows": int(previous_no_shows),
                "appointment_type": str(appointment_type),
                "is_weekend": int(is_weekend),
                "threshold": float(threshold),
            }

            try:
                result = call_api_predict(payload)
                prob = result["no_show_probability"]
                is_flagged = result["predicted_no_show"]
                risk_lvl = result["risk_level"]
                factors = result.get("key_factors", {})

                st.markdown("<br>", unsafe_allow_html=True)
                r_col1, r_col2, r_col3 = st.columns([1, 1, 1.5])

                with r_col1:
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            number={"suffix": "%", "valueformat": ".1f"},
                            title={"text": "No-Show Probability", "font": {"size": 16}},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#2b6cb0"},
                                "steps": [
                                    {"range": [0, 35], "color": "#e6fffa"},
                                    {"range": [35, 65], "color": "#fffaf0"},
                                    {"range": [65, 100], "color": "#fff5f5"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 3},
                                    "thickness": 0.75,
                                    "value": threshold * 100,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with r_col2:
                    badge_class = f"risk-badge-{risk_lvl.lower()}"
                    outcome_text = "Flagged for Potential No-Show" if is_flagged else "Expected Attendance"
                    outcome_color = "#c53030" if is_flagged else "#276749"

                    st.markdown(
                        f"""
                        <div class="metric-card" style="height:230px;">
                            <div class="metric-label">Assessment Result</div>
                            <div style="margin-top:10px;"><span class="{badge_class}">{risk_lvl.upper()} RISK</span></div>
                            <div style="font-size:18px; font-weight:700; color:{outcome_color}; margin-top:14px;">{outcome_text}</div>
                            <div style="font-size:12px; color:#718096; margin-top:8px;">Model: {result.get('model_name', 'Active Pipeline')}</div>
                            <div style="font-size:12px; color:#718096;">Active Threshold: {threshold:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with r_col3:
                    st.markdown(
                        """
                        <div class="metric-card" style="height:230px;">
                            <div class="metric-label">Key Influencing Factors</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    top_drivers = factors.get("top_drivers", [])
                    if top_drivers:
                        for d in top_drivers[:4]:
                            direction_color = "#c53030" if d.get("direction") == "Increases Risk" else "#276749"
                            st.markdown(
                                f"<div style='font-size:13px; margin-top:6px;'>• <strong>{d.get('feature')}</strong>: <span style='color:{direction_color};'>{d.get('direction')}</span> (impact: {d.get('impact'):+.3f})</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown("<div style='font-size:12px; color:#718096; margin-top:10px;'>Standard administrative baseline characteristics applied.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.info(f"Operational Disclaimer: {result.get('disclaimer')}")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

    with tab_batch:
        st.subheader("Batch Appointment Upload & Evaluation")
        st.write("Upload a CSV file containing upcoming scheduled appointments. The model will calculate risk probabilities and produce an actionable list for staff outreach.")

        df_template = pd.DataFrame([
            {
                "lead_time_days": 14,
                "appointment_weekday": "Monday",
                "appointment_month": "October",
                "appointment_hour": 10,
                "booking_channel": "Online",
                "reminder_preference": "SMS",
                "previous_appointments": 4,
                "previous_no_shows": 1,
                "appointment_type": "Specialist",
                "is_weekend": 0,
            },
            {
                "lead_time_days": 2,
                "appointment_weekday": "Wednesday",
                "appointment_month": "October",
                "appointment_hour": 14,
                "booking_channel": "Phone",
                "reminder_preference": "Phone Call",
                "previous_appointments": 2,
                "previous_no_shows": 0,
                "appointment_type": "General Consultation",
                "is_weekend": 0,
            },
        ])

        st.download_button(
            "Download CSV Batch Template",
            data=df_template.to_csv(index=False),
            file_name="hospital_batch_template.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Upload Appointments CSV", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(batch_df)} rows.")

            req_cols = [
                "lead_time_days", "appointment_weekday", "appointment_month",
                "appointment_hour", "booking_channel", "reminder_preference",
                "previous_appointments", "previous_no_shows", "appointment_type", "is_weekend"
            ]
            missing = [c for c in req_cols if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns in CSV: {missing}")
            else:
                model = load_local_model()
                if model is None:
                    st.error("Model artifact not found. Please train the model first.")
                else:
                    batch_probs = model.predict_proba(batch_df[req_cols])[:, 1]
                    batch_df["no_show_probability"] = np.round(batch_probs, 4)
                    batch_df["predicted_no_show"] = (batch_probs >= 0.50).astype(int)

                    def assign_risk(p):
                        if p < 0.35:
                            return "Low"
                        elif p < 0.65:
                            return "Moderate"
                        return "High"

                    batch_df["risk_level"] = batch_df["no_show_probability"].apply(assign_risk)

                    st.markdown("### Batch Evaluation Summary")
                    risk_counts = batch_df["risk_level"].value_counts()
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("High Risk Appointments", int(risk_counts.get("High", 0)))
                    with col_r2:
                        st.metric("Moderate Risk Appointments", int(risk_counts.get("Moderate", 0)))
                    with col_r3:
                        st.metric("Low Risk Appointments", int(risk_counts.get("Low", 0)))

                    st.dataframe(batch_df, use_container_width=True)

                    st.download_button(
                        "Download Prediction Results CSV",
                        data=batch_df.to_csv(index=False),
                        file_name="hospital_predictions_output.csv",
                        mime="text/csv",
                    )


def render_page_model_performance(test_m: dict, comp_df: pd.DataFrame):
    st.markdown('<div class="main-header">Model Performance & Validation Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous Multi-Metric Evaluation on Independent Held-Out Test Datasets</div>', unsafe_allow_html=True)

    if comp_df.empty:
        st.warning("Model comparison data not found. Please run the training pipeline.")
        return

    st.subheader("Candidate Model Comparison (Validation Dataset)")
    st.dataframe(
        comp_df.style.highlight_max(subset=["f1", "roc_auc", "precision", "recall"], color="#e6fffa")
        .highlight_min(subset=["brier_score"], color="#ebf8ff"),
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Held-Out Test Set Metrics (Untouched Test Horizon)")

    test_def = test_m.get("metrics_at_default_threshold", {})
    test_opt = test_m.get("metrics_at_optimal_threshold", {})
    opt_th = test_m.get("optimal_threshold", 0.50)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Precision (Default)", f"{test_def.get('precision', 0):.3f}")
    with col2:
        st.metric("Recall (Default)", f"{test_def.get('recall', 0):.3f}")
    with col3:
        st.metric("F1-Score (Default)", f"{test_def.get('f1', 0):.3f}")
    with col4:
        st.metric("ROC-AUC", f"{test_def.get('roc_auc', 0):.3f}")
    with col5:
        st.metric("Brier Score", f"{test_def.get('brier_score', 0):.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Interactive Decision Threshold Explorer")
    st.write("Examine trade-offs between precision and recall when adjusting the operational decision threshold.")

    thresh_slider = st.slider("Select Operational Threshold", 0.10, 0.90, float(opt_th), step=0.05)

    df_data = load_data()
    model = load_local_model()

    if not df_data.empty and model is not None:
        test_size = int(len(df_data) * 0.15)
        test_subset = df_data.iloc[-test_size:].copy()
        req_cols = [
            "lead_time_days", "appointment_weekday", "appointment_month",
            "appointment_hour", "booking_channel", "reminder_preference",
            "previous_appointments", "previous_no_shows", "appointment_type", "is_weekend"
        ]
        test_probs = model.predict_proba(test_subset[req_cols])[:, 1]
        y_test_true = test_subset["no_show"].to_numpy()

        sweep_df = sweep_thresholds(y_test_true, test_probs)

        fig_tradeoff = px.line(
            sweep_df,
            x="threshold",
            y=["precision", "recall", "f1"],
            title="Precision, Recall, and F1-Score Across Thresholds",
            labels={"threshold": "Classification Threshold", "value": "Metric Score", "variable": "Metric"},
            color_discrete_map={"precision": "#3182ce", "recall": "#dd6b20", "f1": "#38a169"},
        )
        fig_tradeoff.add_vline(x=thresh_slider, line_dash="dash", line_color="red", annotation_text=f"Current: {thresh_slider:.2f}")
        fig_tradeoff.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_tradeoff, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Evaluation Visualizations")
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        roc_path = FIGURES_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curve on Held-Out Test Set")

    with img_col2:
        pr_path = FIGURES_DIR / "precision_recall_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall Curve on Held-Out Test Set")

    img_col3, img_col4 = st.columns(2)
    with img_col3:
        calib_path = FIGURES_DIR / "calibration_curve.png"
        if calib_path.exists():
            st.image(str(calib_path), caption="Reliability / Calibration Curve")

    with img_col4:
        cm_path = FIGURES_DIR / "confusion_matrix_test.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Test Confusion Matrix (Default Threshold)")


def render_page_data_explorer(df: pd.DataFrame):
    st.markdown('<div class="main-header">Appointment Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Querying, Filtering, and Inspection of Synthetic Hospital Appointments</div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("Dataset not found. Please run the data generation pipeline.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        channel_filter = st.multiselect("Booking Channel", options=df["booking_channel"].dropna().unique(), default=list(df["booking_channel"].dropna().unique()))
    with col2:
        type_filter = st.multiselect("Appointment Type", options=df["appointment_type"].dropna().unique(), default=list(df["appointment_type"].dropna().unique()))
    with col3:
        status_filter = st.selectbox("Attendance Status", ["All", "Attended Only (0)", "Missed Only (1)"])
    with col4:
        search_id = st.text_input("Search Appointment ID", "")

    filtered = df.copy()
    if channel_filter:
        filtered = filtered[filtered["booking_channel"].isin(channel_filter)]
    if type_filter:
        filtered = filtered[filtered["appointment_type"].isin(type_filter)]
    if status_filter == "Attended Only (0)":
        filtered = filtered[filtered["no_show"] == 0]
    elif status_filter == "Missed Only (1)":
        filtered = filtered[filtered["no_show"] == 1]
    if search_id.strip():
        filtered = filtered[filtered["appointment_id"].str.contains(search_id.strip(), case=False, na=False)]

    st.write(f"Showing **{len(filtered):,}** of **{len(df):,}** records")
    st.dataframe(filtered.head(500), use_container_width=True)

    st.download_button(
        "Download Filtered Dataset as CSV",
        data=filtered.to_csv(index=False),
        file_name="hospital_appointments_filtered.csv",
        mime="text/csv",
    )


def render_page_explainability(meta: dict):
    st.markdown('<div class="main-header">Model Explainability & Feature Drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transparent Inspection of Global and Local Feature Contributions</div>', unsafe_allow_html=True)

    feat_path = FIGURES_DIR / "feature_importance.png"
    if feat_path.exists():
        st.image(str(feat_path), caption="Global Relative Feature Importances", use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Key Empirical Insights from Model Features")

    st.markdown(
        """
        - **Lead Time (Days):** The single strongest administrative driver. Longer lead times consistently elevate no-show risk due to scheduling horizon decay, patient schedule volatility, and resolution of symptoms before consultation.
        - **Historical No-Show Ratio:** Prior attendance track record provides a strong baseline behavioral signal. Patients with multiple prior missed appointments exhibit significantly higher likelihood of future missed visits.
        - **Reminder Preference & Outreach Channel:** Automated SMS and active phone call reminders correlate with higher attendance compliance relative to passive or unspecified reminder settings.
        - **Booking Channel:** In-person and clinician referrals demonstrate lower no-show rates compared to generic telephone bookings, reflecting higher patient commitment at the point of scheduling.
        - **Timing Effects:** Weekend slots and late afternoon appointments show modest increases in no-show propensity.
        """
    )

    st.info(
        "Scientific Attribution Note: Feature importances reflect statistical associations within the trained model and administrative features. They do not represent medical diagnoses, clinical determinations, or strict causal claims."
    )


def render_page_about():
    st.markdown('<div class="main-header">About the System & Responsible AI Principles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Governance, Anti-Leakage Architecture, and Operational Ethics</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 1. Project Objective & Operational Context
        The **Hospital Appointment No-Show Prediction System** is an AI-driven operational decision-support tool engineered for hospital administrators, front-desk coordinators, and clinical scheduling directors. Its purpose is to quantify the probability of appointment non-attendance at the time of booking, enabling preemptive interventions such as targeted SMS reminders, transportation assistance, or buffer scheduling.

        ### 2. Strict Anti-Leakage Protocol
        Predictions are made strictly at the **point of appointment booking**. The model incorporates only administrative and scheduling information known at booking time:
        - **Zero Post-Booking Information:** No post-booking attendance updates, reminder delivery confirmations, cancellations, or clinic check-in events are accessible to the model.
        - **Zero Sensitive Clinical Data:** The system does not access patient diagnoses, clinical notes, treatment protocols, surgical histories, or laboratory results.
        - **Strict Historical Boundaries:** `previous_no_shows` is strictly bounded by `previous_appointments`, capturing past behavior without target leakage.

        ### 3. Chronological Validation Strategy
        To prevent lookahead bias and temporal leakage, the dataset is split chronologically into:
        - **Training Set (70%):** Past appointments used exclusively to fit estimators and preprocessors.
        - **Validation Set (15%):** Intermediate appointments used for hyperparameter tuning, model comparison, threshold selection, and calibration analysis.
        - **Held-Out Test Set (15%):** Future appointment horizon untouched until final benchmark reporting.

        ### 4. Ethical AI and Human-in-the-Loop Governance
        - **Decision-Support Only:** The system's output is an advisory risk score, not an automated judgment. It must never be used to deny, deprioritize, or cancel patient care.
        - **Non-Discriminatory Scheduling:** High-risk predictions should trigger patient support (e.g., proactive reminder outreach or assistance), never penalty fees or care deprioritization.
        - **Synthetic Data Declaration:** This demonstration environment uses a controlled synthetic dataset designed to mirror realistic administrative distributions. It should never be cited as real-world patient outcomes.
        """
    )


def main():
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            [
                "Overview Dashboard",
                "Predict Appointment",
                "Model Performance",
                "Data Explorer",
                "Model Explainability",
                "About & Responsible AI",
            ],
            index=0,
        )

        st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:12px; color:#718096;">
                <strong>Algions AI Labs</strong><br>
                Healthcare Intelligence Division<br>
                Status: <span style="color:#38a169; font-weight:bold;">Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    df = load_data()
    meta = load_metadata()
    test_m = load_test_metrics()
    comp_df = load_comparison_metrics()

    if page == "Overview Dashboard":
        render_page_overview(df, meta, test_m)
    elif page == "Predict Appointment":
        render_page_predict()
    elif page == "Model Performance":
        render_page_model_performance(test_m, comp_df)
    elif page == "Data Explorer":
        render_page_data_explorer(df)
    elif page == "Model Explainability":
        render_page_explainability(meta)
    elif page == "About & Responsible AI":
        render_page_about()


if __name__ == "__main__":
    main()
