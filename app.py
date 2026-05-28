import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from groq import Groq
import os

# =============================================================================
# GROQ API KEY  — paste your key here
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")   # ← replace with your actual key

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title = "Patient Readmission Prediction App",
    page_icon  = "🏥",
    layout     = "wide"
)

# =============================================================================
# LOAD AND PREPARE DATA  (cached so it only runs once)
# =============================================================================

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("patient_data.csv")

    df = df.drop(columns=["A1Cresult", "max_glu_serum"])
    df = df.drop(columns=["admission_type_id",
                           "discharge_disposition_id",
                           "admission_source_id"])

    df["race"]      = df["race"].fillna(df["race"].mode()[0])
    df              = df[df["gender"] != "Unknown/Invalid"]

    age_map = {
        "[0-10)": 0,  "[10-20)": 10, "[20-30)": 20, "[30-40)": 30,
        "[40-50)": 40, "[50-60)": 50, "[60-70)": 60, "[70-80)": 70,
        "[80-90)": 80, "[90-100)": 90
    }
    df["age_numeric"]     = df["age"].map(age_map)
    df["insulin_numeric"] = df["insulin"].map({"No": 0, "Down": 1, "Steady": 2, "Up": 3})
    df["diabetesMed"]     = df["diabetesMed"].map({"Yes": 1, "No": 0})
    df["change"]          = df["change"].map({"Ch": 1, "No": 0})
    df["gender_num"]      = df["gender"].map({"Male": 1, "Female": 0})

    med_cols = ["metformin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone"]
    df["total_meds_active"] = 0
    for col in med_cols:
        df["total_meds_active"] += (df[col] != "No").astype(int)
    df = df.drop(columns=med_cols)

    df["service_intensity"] = (df["time_in_hospital"] +
                                df["number_inpatient"]  +
                                df["number_outpatient"] +
                                df["number_emergency"])

    df = pd.get_dummies(df, columns=["race"], drop_first=True, dtype=int)
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

    return df


@st.cache_resource
def train_model(df):
    feature_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses", "gender_num",
        "diabetesMed", "change", "age_numeric", "total_meds_active",
        "insulin_numeric", "service_intensity"
    ]
    race_cols = [c for c in df.columns if c.startswith("race_")]
    feature_cols += race_cols

    X = df[feature_cols]
    y = df["readmitted_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # keep training stats so we can compare a new patient against averages
    feature_stats = X_train.describe()

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators = 100,
        max_depth    = 10,
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1
    )
    model.fit(X_train_sc, y_train)

    return model, scaler, feature_cols, feature_stats


# =============================================================================
# LOAD DATA AND MODEL
# =============================================================================

df                                        = load_and_prepare_data()
model, scaler, feature_cols, feature_stats = train_model(df)


# =============================================================================
# HELPER : compute top contributing factors for a single patient
# =============================================================================

def get_top_factors(input_row_df, feature_cols, model, feature_stats, top_n=7):
    """
    For each feature we compute:
        contribution = global_feature_importance × |patient_value - dataset_mean| / std

    This combines how important a feature is globally with how extreme
    this particular patient's value is compared to the average patient.
    A high score means the feature is both important AND unusual for this patient.
    """
    importances = model.feature_importances_
    factors = []

    for i, feat in enumerate(feature_cols):
        patient_val = float(input_row_df[feat].values[0])
        avg_val     = float(feature_stats.loc["mean", feat])
        std_val     = float(feature_stats.loc["std",  feat])

        deviation   = (patient_val - avg_val) / (std_val + 1e-9)
        score       = importances[i] * abs(deviation)

        if deviation > 0.2:
            direction = "above average"
        elif deviation < -0.2:
            direction = "below average"
        else:
            direction = "near average"

        factors.append({
            "feature"   : feat,
            "value"     : patient_val,
            "average"   : round(avg_val, 2),
            "direction" : direction,
            "score"     : score,
            "importance": round(importances[i], 4)
        })

    factors.sort(key=lambda x: x["score"], reverse=True)
    return factors[:top_n]


# =============================================================================
# HELPER : call Claude to generate a clinical summary
# =============================================================================

def generate_llm_summary(api_key, patient_info, prediction_result,
                          prob_positive, top_factors):
    """
    Builds a structured prompt with the patient details, model prediction,
    and top contributing factors, then sends it to Groq (llama-3.3-70b-versatile)
    and returns the response as a string.
    """

    # build readable list of top factors
    factors_text = ""
    for i, f in enumerate(top_factors, 1):
        factors_text += (
            f"  {i}. {f['feature']} — "
            f"patient value: {f['value']}, "
            f"dataset average: {f['average']}, "
            f"direction: {f['direction']}, "
            f"global feature importance: {f['importance']}\n"
        )

    prompt = f"""
You are a clinical decision-support assistant. A Random Forest machine learning model
has analysed a diabetic patient's hospital record and predicted their readmission risk.

Write a clear and concise clinical summary for a healthcare professional.

---
PATIENT DETAILS:
{patient_info}

---
MODEL PREDICTION:
- Outcome: {"HIGH RISK — Likely to be readmitted within 30 days" if prediction_result == 1 else "LOW RISK — Not likely to be readmitted within 30 days"}
- Readmission probability (within 30 days): {prob_positive:.1f}%

---
TOP CONTRIBUTING FACTORS (ranked by how much they influenced this specific prediction):
{factors_text}

---
Write a clinical summary with three short paragraphs:

Paragraph 1 — Verdict: One or two sentences stating the patient's readmission risk level
and the model's confidence (probability).

Paragraph 2 — Key drivers: Explain the 3 or 4 most important factors that drove this
prediction, in plain clinical language. Use readable labels like "number of inpatient
visits" instead of variable names like "number_inpatient". Mention whether each factor
is above or below average for this patient and why that matters for readmission risk.

Paragraph 3 — Clinical note: One or two sentences on what the care team might want
to watch or act on based on this risk profile.

Keep the total response under 220 words. Write in paragraph form, not bullet points.
Be factual and do not speculate beyond what the data shows.
"""

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model      = "llama-3.3-70b-versatile",
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = 400
    )

    return response.choices[0].message.content


# =============================================================================
# HELPER : build the system prompt for the Q&A chatbot
# =============================================================================

def build_chat_system_prompt(patient_info, prediction_result,
                              prob_positive, top_factors):
    """
    Creates a rich system prompt that gives the chatbot full context about
    the current patient so it can answer any follow-up question accurately.
    The system prompt is sent once at the start of every API call as the
    first message with role='system'.
    """

    # convert top factors to a readable list for the prompt
    factors_text = ""
    for i, f in enumerate(top_factors, 1):
        label = LABEL_MAP.get(f["feature"], f["feature"])
        factors_text += (
            f"  {i}. {label}: patient value = {f['value']}, "
            f"dataset average = {f['average']}, {f['direction']}\n"
        )

    system_prompt = f"""
You are a clinical decision-support assistant embedded in a patient readmission
prediction tool. You have full context about the current patient and their
readmission risk prediction from a Random Forest model.

Your job is to answer questions from healthcare professionals clearly, accurately,
and concisely. Use plain clinical language. Do not speculate beyond what the data
shows. If asked something outside your knowledge, say so honestly.

---
CURRENT PATIENT DETAILS:
{patient_info}

---
MODEL PREDICTION:
- Risk level: {"HIGH RISK — likely readmitted within 30 days" if prediction_result == 1 else "LOW RISK — not likely readmitted within 30 days"}
- Readmission probability (within 30 days): {prob_positive:.1f}%
- Model used: Random Forest (100 trees, balanced class weights)

---
TOP FACTORS DRIVING THIS PREDICTION:
{factors_text}

---
GUIDELINES FOR YOUR RESPONSES:
- Keep answers under 120 words unless the question genuinely requires more detail.
- When explaining a feature, use the readable label (e.g. "number of inpatient visits")
  not the variable name (e.g. "number_inpatient").
- If asked what would lower the risk, focus on the top contributing factors that
  are clinically actionable (e.g. reducing inpatient visits, adjusting medications).
- If asked about the model itself, explain it simply: a Random Forest trains 100
  decision trees on patient data and combines their votes.
- Never make definitive diagnoses or treatment decisions.
"""
    return system_prompt.strip()


# =============================================================================
# READABLE FEATURE LABEL MAP (used in charts and prompt)
# =============================================================================

LABEL_MAP = {
    "time_in_hospital"  : "Days in Hospital",
    "num_lab_procedures": "Lab Procedures",
    "num_procedures"    : "Medical Procedures",
    "num_medications"   : "Number of Medications",
    "number_outpatient" : "Outpatient Visits",
    "number_emergency"  : "Emergency Visits",
    "number_inpatient"  : "Inpatient Visits",
    "number_diagnoses"  : "Number of Diagnoses",
    "gender_num"        : "Gender",
    "diabetesMed"       : "On Diabetes Medication",
    "change"            : "Medication Changed",
    "age_numeric"       : "Patient Age",
    "total_meds_active" : "Active Oral Medications",
    "insulin_numeric"   : "Insulin Usage Level",
    "service_intensity" : "Service Intensity Score",
    "race_Asian"        : "Race: Asian",
    "race_Caucasian"    : "Race: Caucasian",
    "race_Hispanic"     : "Race: Hispanic",
    "race_Other"        : "Race: Other",
}


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

# st.sidebar.image(
#     "https://img.icons8.com/ios-filled/100/4A90D9/hospital.png",
#     width=60
# )
st.sidebar.title("Patient Readmission")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["Data Dashboard", "Predict Readmission"]
)

st.sidebar.markdown("---")
# # if llm_enabled:
# #     st.sidebar.success("🤖 AI summary enabled")
# # else:
# #     st.sidebar.warning("Set GROQ_API_KEY in app.py to enable AI summary")

# st.sidebar.markdown("---")
# st.sidebar.markdown(
#     f"**Dataset:** {len(df):,} patients  \n"
#     f"**Features:** {len(feature_cols)}  \n"
#     f"**Model:** Random Forest  \n"
#     f"**Positive class:** {df['readmitted_binary'].mean()*100:.1f}%"
# )

# API key is set as a constant at the top of the file — no UI input needed
api_key_input = GROQ_API_KEY
llm_enabled   = bool(GROQ_API_KEY and GROQ_API_KEY != "your-groq-api-key-here")


# =============================================================================
# PAGE 1 : DATA DASHBOARD
# =============================================================================

if page == "Data Dashboard":

    st.title("🏥 Patient Readmission — Data Dashboard")
    st.markdown("Explore the diabetic patient dataset used to train the readmission prediction model.")
    st.markdown("---")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients",  f"{len(df):,}")
    c2.metric("Readmitted <30d", f"{df['readmitted_binary'].sum():,}")
    c3.metric("Not Readmitted",  f"{(df['readmitted_binary']==0).sum():,}")
    c4.metric("Positive Rate",   f"{df['readmitted_binary'].mean()*100:.1f}%")
    st.markdown("---")

    # ── Target distribution + gender ─────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Readmission class distribution")
        counts = df["readmitted"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        fig1 = px.pie(counts, names="Class", values="Count",
                      color_discrete_sequence=["#4CAF50", "#FF9800", "#F44336"],
                      hole=0.45)
        fig1.update_traces(textposition="outside", textinfo="percent+label")
        fig1.update_layout(showlegend=True, margin=dict(t=20,b=20,l=20,r=20), height=340)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.subheader("Readmission rate by gender")
        gr = df.groupby("gender")["readmitted_binary"].mean().reset_index()
        gr.columns = ["Gender", "Rate"]
        # gr["Gender"]  = gr["Gender"].map({1: "Male", 0: "Female"})
        gr["Rate %"]  = (gr["Rate"] * 100).round(2)
        fig2 = px.bar(gr, x="Gender", y="Rate %", color="Gender",
                      color_discrete_sequence=["#378ADD","#D4537E"], text="Rate %")
        fig2.update_traces(texttemplate="%{text:.2f}%", textposition="inside")
        fig2.update_layout(showlegend=False, yaxis_title="Readmission Rate (%)",
                           margin=dict(t=20,b=20,l=20,r=20), height=340)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Age + time in hospital ────────────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Readmission rate by age group")
        age_labels = {0:"0-10",10:"10-20",20:"20-30",30:"30-40",40:"40-50",
                      50:"50-60",60:"60-70",70:"70-80",80:"80-90",90:"90-100"}
        adf = df.copy()
        adf["Age Group"] = adf["age_numeric"].map(age_labels)
        ar = adf.groupby("Age Group")["readmitted_binary"].mean().reset_index()
        ar.columns = ["Age Group", "Rate"]
        ar["Rate %"] = (ar["Rate"] * 100).round(2)
        order = ["0-10","10-20","20-30","30-40","40-50",
                 "50-60","60-70","70-80","80-90","90-100"]
        ar["Age Group"] = pd.Categorical(ar["Age Group"], categories=order, ordered=True)
        ar = ar.sort_values("Age Group")
        fig3 = px.bar(ar, x="Age Group", y="Rate %",
                      color="Rate %", color_continuous_scale="Blues", text="Rate %")
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.update_layout(coloraxis_showscale=False, yaxis_title="Readmission Rate (%)",
                           margin=dict(t=20,b=20,l=20,r=20), height=340)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.subheader("Time in hospital distribution")
        fig4 = px.histogram(df, x="time_in_hospital", color="readmitted_binary",
                            barmode="overlay", nbins=14,
                            labels={"readmitted_binary":"Readmitted <30d",
                                    "time_in_hospital":"Days in Hospital"},
                            color_discrete_map={0:"#4CAF50", 1:"#F44336"})
        fig4.update_layout(yaxis_title="Number of Patients",
                           margin=dict(t=20,b=20,l=20,r=20), height=340,
                           legend=dict(title="Readmitted <30d", x=0.75, y=0.95))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Race + medications ────────────────────────────────────────────────────
    col_e, col_f = st.columns(2)

    with col_e:
        st.subheader("Readmission rate by race")
        race_dummies = [c for c in df.columns if c.startswith("race_")]
        race_rates   = {}
        for col in race_dummies:
            name = col.replace("race_", "")
            mask = df[col] == 1
            if mask.sum() > 0:
                race_rates[name] = df.loc[mask, "readmitted_binary"].mean() * 100
        base_mask = (df[race_dummies] == 0).all(axis=1)
        race_rates["AfricanAmerican"] = df.loc[base_mask, "readmitted_binary"].mean() * 100
        rdf = pd.DataFrame(list(race_rates.items()), columns=["Race", "Rate %"])
        rdf = rdf.sort_values("Rate %", ascending=True)
        fig5 = px.bar(rdf, x="Rate %", y="Race", orientation="h",
                      color="Rate %", color_continuous_scale="Teal", text="Rate %")
        fig5.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig5.update_layout(coloraxis_showscale=False, xaxis_title="Readmission Rate (%)",
                           margin=dict(t=20,b=20,l=20,r=20), height=340)
        st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        st.subheader("Number of medications vs readmission rate")
        mr = df.groupby("num_medications")["readmitted_binary"].mean().reset_index()
        mr.columns = ["Num Medications", "Rate"]
        mr["Rate %"] = (mr["Rate"] * 100).round(2)
        fig6 = px.scatter(mr, x="Num Medications", y="Rate %",
                          size="Rate %", color="Rate %",
                          color_continuous_scale="Reds",
                          labels={"Rate %": "Readmission Rate (%)"})
        fig6.update_layout(coloraxis_showscale=False, yaxis_title="Readmission Rate (%)",
                           margin=dict(t=20,b=20,l=20,r=20), height=340)
        st.plotly_chart(fig6, use_container_width=True)
    # ── Inpatient visits + Lab procedures ────────────────────────────────────
    col_g, col_h = st.columns(2)
 
    with col_g:
     st.subheader("Readmission rate by inpatient visits")
     st.caption("Number of inpatient visits in the year before this admission — the model's top feature")
     
     # Group visit counts: 0, 1, 2, 3, 4, 5+ to keep the chart readable
     idf = df.copy()
     idf["Inpatient Visits"] = idf["number_inpatient"].apply(
     lambda x: "5+" if x >= 5 else str(int(x))
     )
     ip = idf.groupby("Inpatient Visits")["readmitted_binary"].agg(
     Rate="mean", Count="count"
     ).reset_index()
     ip["Rate %"] = (ip["Rate"] * 100).round(2)
     
     # keep natural order 0,1,2,3,4,5+
     visit_order = ["0","1","2","3","4","5+"]
     ip["Inpatient Visits"] = pd.Categorical(
     ip["Inpatient Visits"], categories=visit_order, ordered=True
     )
     ip = ip.sort_values("Inpatient Visits")
     
     fig_ip = px.bar(
     ip, x="Inpatient Visits", y="Rate %",
     color="Rate %", color_continuous_scale="Oranges",
     text="Rate %",
     labels={"Rate %": "Readmission Rate (%)"}
     )
     fig_ip.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
     fig_ip.update_layout(
     coloraxis_showscale=False,
     yaxis_title="Readmission Rate (%)",
     xaxis_title="Number of Inpatient Visits",
     margin=dict(t=20,b=20,l=20,r=20),
     height=360
     )
     st.plotly_chart(fig_ip, use_container_width=True)
     
    with col_h:
     st.subheader("Readmission rate by lab procedures")
     st.caption("Number of lab tests performed — second most important feature in the model")
     
     # Bin into ranges so the chart doesn't have 130 bars
     ldf = df.copy()
     bins = [0, 20, 40, 60, 80, 100, 135]
     labels = ["1–20","21–40","41–60","61–80","81–100","100+"]
     ldf["Lab Procedures"] = pd.cut(
     ldf["num_lab_procedures"], bins=bins, labels=labels, right=True
     )
     lp = ldf.groupby("Lab Procedures", observed=True)["readmitted_binary"].agg(
     Rate="mean", Count="count"
     ).reset_index()
     lp["Rate %"] = (lp["Rate"] * 100).round(2)
     
     fig_lp = px.bar(
     lp, x="Lab Procedures", y="Rate %",
     color="Rate %", color_continuous_scale="Purples",
     text="Rate %",
     labels={"Rate %": "Readmission Rate (%)"}
     )
     fig_lp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
     fig_lp.update_layout(
     coloraxis_showscale=False,
     yaxis_title="Readmission Rate (%)",
     xaxis_title="Number of Lab Procedures",
     margin=dict(t=20,b=20,l=20,r=20),
     height=360
     )
     st.plotly_chart(fig_lp, use_container_width=True)
 

    # ── Feature importance ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Top 15 feature importances (Random Forest)")
    imp_df = pd.DataFrame({
        "Feature":    [LABEL_MAP.get(f, f) for f in feature_cols],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)
    fig7 = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature",
                  orientation="h", color="Importance", color_continuous_scale="Blues")
    fig7.update_layout(coloraxis_showscale=False, xaxis_title="Importance Score",
                       margin=dict(t=10,b=20,l=10,r=10), height=450)
    st.plotly_chart(fig7, use_container_width=True)


# =============================================================================
# PAGE 2 : PREDICTION
# =============================================================================

elif page == "Predict Readmission":

    st.title("Predict Patient Readmission")
    st.markdown(
        "Fill in the patient details below and click **Predict**. "
        "The model will estimate readmission risk and "
        "the AI assistant will generate a clinical summary explaining the prediction."
    )
    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):

        st.subheader("Patient demographics")
        d1, d2, d3 = st.columns(3)
        with d1:
            age_input = st.selectbox(
                "Age group",
                ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                 "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"],
                index=6
            )
        with d2:
            gender_input = st.selectbox("Gender", ["Female", "Male"])
        with d3:
            race_input = st.selectbox(
                "Race",
                ["Caucasian","AfricanAmerican","Hispanic","Asian","Other"]
            )

        st.markdown("---")
        st.subheader("Hospital visit details")
        h1, h2, h3, h4 = st.columns(4)
        with h1: time_in_hospital   = st.slider("Days in hospital",    1, 14, 4)
        with h2: num_lab_procedures = st.slider("Lab procedures",      1, 132, 43)
        with h3: num_procedures     = st.slider("Medical procedures",  0, 6, 1)
        with h4: num_medications    = st.slider("Medications",         1, 81, 15)

        h5, h6, h7, h8 = st.columns(4)
        with h5: number_outpatient = st.slider("Outpatient visits",   0, 42, 0)
        with h6: number_emergency  = st.slider("Emergency visits",    0, 76, 0)
        with h7: number_inpatient  = st.slider("Inpatient visits",    0, 21, 0)
        with h8: number_diagnoses  = st.slider("Number of diagnoses", 1, 16, 7)

        st.markdown("---")
        st.subheader("Medication details")
        m1, m2, m3 = st.columns(3)
        with m1: insulin_input = st.selectbox("Insulin usage",           ["No","Down","Steady","Up"])
        with m2: diabetes_med  = st.selectbox("On diabetes medication?", ["Yes","No"])
        with m3: change_med    = st.selectbox("Medication changed?",     ["Yes","No"])

        st.markdown("**Oral medications the patient is taking**")
        o1, o2, o3, o4, o5 = st.columns(5)
        met  = o1.selectbox("Metformin",     ["No","Steady","Up","Down"])
        glip = o2.selectbox("Glipizide",     ["No","Steady","Up","Down"])
        glyb = o3.selectbox("Glyburide",     ["No","Steady","Up","Down"])
        pio  = o4.selectbox("Pioglitazone",  ["No","Steady","Up","Down"])
        rosi = o5.selectbox("Rosiglitazone", ["No","Steady","Up","Down"])

        st.markdown("---")
        submitted = st.form_submit_button("Predict", use_container_width=True)

    # ── COMPUTE : runs only when Predict button is clicked ───────────────────
    if submitted:

        age_map_num     = {"[0-10)":0,"[10-20)":10,"[20-30)":20,"[30-40)":30,
                           "[40-50)":40,"[50-60)":50,"[60-70)":60,"[70-80)":70,
                           "[80-90)":80,"[90-100)":90}
        insulin_num_map = {"No":0,"Down":1,"Steady":2,"Up":3}

        age_numeric       = age_map_num[age_input]
        gender_num        = 1 if gender_input == "Male" else 0
        insulin_numeric   = insulin_num_map[insulin_input]
        diabetes_num      = 1 if diabetes_med == "Yes" else 0
        change_num        = 1 if change_med   == "Yes" else 0
        meds_active       = sum([m != "No" for m in [met, glip, glyb, pio, rosi]])
        service_intensity = (time_in_hospital + number_inpatient +
                             number_outpatient + number_emergency)

        input_data = {
            "time_in_hospital"  : time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures"    : num_procedures,
            "num_medications"   : num_medications,
            "number_outpatient" : number_outpatient,
            "number_emergency"  : number_emergency,
            "number_inpatient"  : number_inpatient,
            "number_diagnoses"  : number_diagnoses,
            "gender_num"        : gender_num,
            "diabetesMed"       : diabetes_num,
            "change"            : change_num,
            "age_numeric"       : age_numeric,
            "total_meds_active" : meds_active,
            "insulin_numeric"   : insulin_numeric,
            "service_intensity" : service_intensity,
            "race_Asian"        : 1 if race_input == "Asian"     else 0,
            "race_Caucasian"    : 1 if race_input == "Caucasian" else 0,
            "race_Hispanic"     : 1 if race_input == "Hispanic"  else 0,
            "race_Other"        : 1 if race_input == "Other"     else 0,
        }

        input_df     = pd.DataFrame([input_data])[feature_cols]
        input_scaled = scaler.transform(input_df)

        prediction    = model.predict(input_scaled)[0]
        probability   = model.predict_proba(input_scaled)[0]
        prob_positive = probability[1] * 100
        prob_negative = probability[0] * 100
        top_factors   = get_top_factors(input_df, feature_cols, model, feature_stats)

        patient_info_str = (
            f"Age group: {age_input}\n"
            f"Gender: {gender_input}\n"
            f"Race: {race_input}\n"
            f"Days in hospital: {time_in_hospital}\n"
            f"Number of lab procedures: {num_lab_procedures}\n"
            f"Number of medical procedures: {num_procedures}\n"
            f"Number of medications: {num_medications}\n"
            f"Outpatient visits (past year): {number_outpatient}\n"
            f"Emergency visits (past year): {number_emergency}\n"
            f"Inpatient visits (past year): {number_inpatient}\n"
            f"Number of diagnoses: {number_diagnoses}\n"
            f"Insulin usage: {insulin_input}\n"
            f"On diabetes medication: {diabetes_med}\n"
            f"Medication changed this visit: {change_med}\n"
            f"Active oral diabetes drugs: {meds_active} out of 5 "
            f"(metformin={met}, glipizide={glip}, glyburide={glyb}, "
            f"pioglitazone={pio}, rosiglitazone={rosi})\n"
            f"Service intensity score: {service_intensity}"
        )

        # Generate AI summary once at predict time and cache it
        ai_summary = None
        if llm_enabled:
            try:
                ai_summary = generate_llm_summary(
                    api_key           = api_key_input,
                    patient_info      = patient_info_str,
                    prediction_result = prediction,
                    prob_positive     = prob_positive,
                    top_factors       = top_factors
                )
            except Exception as e:
                ai_summary = f"Could not generate summary: {e}"

        # Save everything to session_state so the render block below
        # can draw all outputs on every rerun — including chat reruns
        st.session_state.pred = {
            "prediction"      : int(prediction),
            "prob_positive"   : prob_positive,
            "prob_negative"   : prob_negative,
            "top_factors"     : top_factors,
            "patient_info_str": patient_info_str,
            "ai_summary"      : ai_summary,
            # summary card values
            "age_input"           : age_input,
            "gender_input"        : gender_input,
            "race_input"          : race_input,
            "time_in_hospital"    : time_in_hospital,
            "num_lab_procedures"  : num_lab_procedures,
            "num_medications"     : num_medications,
            "number_diagnoses"    : number_diagnoses,
            "service_intensity"   : service_intensity,
            "insulin_input"       : insulin_input,
            "diabetes_med"        : diabetes_med,
            "change_med"          : change_med,
            "meds_active"         : meds_active,
        }

        # Reset chat history for the new patient
        st.session_state.chat_history    = []
        st.session_state.patient_context = patient_info_str
        st.session_state.top_factors_ctx = top_factors
        st.session_state.prediction_ctx  = int(prediction)
        st.session_state.prob_ctx        = prob_positive

    # ── RENDER : runs on every rerun as long as a prediction exists ──────────
    # This block is OUTSIDE if submitted so it persists through chat reruns,
    # button clicks, and any other interaction that triggers a Streamlit rerun.
    if "pred" not in st.session_state:
        st.info("Fill in the form above and click **Predict** to see results.")

    else:
        p = st.session_state.pred   # short alias for readability

        # ── Result banner ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Prediction result")

        if p["prediction"] == 1:
            st.error(
                f"⚠️  **High Risk — Patient is likely to be readmitted within 30 days**  \n"
                f"Readmission probability: **{p['prob_positive']:.1f}%**"
            )
        else:
            st.success(
                f"✅  **Low Risk — Patient is NOT likely to be readmitted within 30 days**  \n"
                f"Readmission probability: **{p['prob_positive']:.1f}%**"
            )

        # ── Gauge + probability split ─────────────────────────────────────────
        g1, g2 = st.columns(2)

        with g1:
            gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = round(p["prob_positive"], 1),
                title = {"text": "Readmission Risk (%)", "font": {"size": 16}},
                # delta = {"reference": 11.2,
                #          "increasing": {"color": "#F44336"},
                #          "decreasing": {"color": "#4CAF50"}},
                gauge = {
                    "axis"  : {"range": [0, 100]},
                    "bar"   : {"color": "#F44336" if p["prediction"] == 1 else "#4CAF50"},
                    "steps" : [
                        {"range": [0,  30], "color": "#E8F5E9"},
                        {"range": [30, 60], "color": "#FFF9C4"},
                        {"range": [60, 100],"color": "#FFEBEE"},
                    ],
                    "threshold": {"line": {"color": "#333", "width": 3},
                                  "thickness": 0.8, "value": 50}
                }
            ))
            gauge.update_layout(height=280, margin=dict(t=30,b=10,l=30,r=30))
            st.plotly_chart(gauge, use_container_width=True)

        with g2:
            pf = go.Figure()
            pf.add_trace(go.Bar(
                x=[p["prob_negative"]], y=[""], orientation="h",
                name="Not readmitted", marker_color="#4CAF50",
                text=f"{p['prob_negative']:.2f}%", textposition="inside"
            ))
            pf.add_trace(go.Bar(
                x=[p["prob_positive"]], y=[""], orientation="h",
                name="Readmitted <30d", marker_color="#F44336",
                text=f"{p['prob_positive']:.2f}%", textposition="inside"
            ))
            pf.update_layout(
                barmode="stack", height=280, title="Probability breakdown",
                xaxis=dict(range=[0,100], title="Probability (%)"),
                margin=dict(t=50,b=30,l=10,r=10),
                legend=dict(orientation="h", y=1.1, x=0.6)
            )
            st.plotly_chart(pf, use_container_width=True)

        # ── Top contributing factors chart ─────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Top factors contributing to this prediction")
        st.caption(
            "Each bar = global feature importance × how far this patient deviates "
            "from the dataset average."
        )

        fdf = pd.DataFrame(p["top_factors"])
        fdf["label"]     = fdf["feature"].map(LABEL_MAP).fillna(fdf["feature"])
        fdf["value_str"] = fdf.apply(
            lambda r: f"Patient: {r['value']:.1f}  |  Avg: {r['average']:.1f}  ({r['direction']})",
            axis=1
        )
        fig_f = px.bar(
            fdf.sort_values("score"),
            x="score", y="label", orientation="h",
            color="score", color_continuous_scale="RdYlGn_r",
            text="value_str",
            labels={"score": "Contribution Score", "label": "Feature"}
        )
        fig_f.update_traces(textposition="outside", textfont_size=10)
        fig_f.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Contribution Score (importance × deviation from average)",
            margin=dict(t=10,b=10,l=10,r=220),
            height=380
        )
        st.plotly_chart(fig_f, use_container_width=True)

        # ── Patient summary cards ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Patient summary")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**Demographics**")
            st.write(f"Age group: {p['age_input']}")
            st.write(f"Gender: {p['gender_input']}")
            st.write(f"Race: {p['race_input']}")
        with s2:
            st.markdown("**Hospital visit**")
            st.write(f"Days in hospital: {p['time_in_hospital']}")
            st.write(f"Lab procedures: {p['num_lab_procedures']}")
            st.write(f"Medications: {p['num_medications']}")
            st.write(f"Diagnoses: {p['number_diagnoses']}")
            st.write(f"Service intensity score: {p['service_intensity']}")
        with s3:
            st.markdown("**Medication**")
            st.write(f"Insulin: {p['insulin_input']}")
            st.write(f"On diabetes med: {p['diabetes_med']}")
            st.write(f"Medication changed: {p['change_med']}")
            st.write(f"Active oral drugs: {p['meds_active']} / 5")

        # ── AI Clinical Summary ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🤖 AI Clinical Summary")

        if not llm_enabled:
            st.info(
                "Set your Groq API key in the `GROQ_API_KEY` constant "
                "at the top of app.py to enable the AI clinical summary."
            )
        elif p["ai_summary"]:
            st.markdown(
                f"""<div style="background-color:#f0f4ff;border-left:4px solid #378ADD;
                border-radius:6px;padding:1rem 1.25rem;font-size:15px;
                line-height:1.8;color:#1a1a2e;">
                {p['ai_summary'].replace(chr(10), "<br>")}
                </div>""",
                unsafe_allow_html=True
            )
            st.caption(
                "For clinical decision-support only — does not replace medical judgement."
            )

    # =========================================================================
    # PATIENT Q&A CHATBOT
    # Appears below the prediction after the first Predict click.
    # Persists across reruns via st.session_state so the conversation
    # stays alive while the user types follow-up questions.
    # =========================================================================

    if "patient_context" not in st.session_state:
        # No prediction made yet — show a prompt to run prediction first
        st.info("Run a prediction above to activate the patient Q&A assistant.")

    else:
        st.markdown("---")
        st.subheader("💬 Patient Q&A Assistant")
        st.caption(
            "Ask any follow-up question about this patient's prediction — "
            "e.g. *'Why is this patient high risk?'*, *'What could lower the risk?'*, "
            "*'What does service intensity mean?'*"
        )

        # ── Clear chat button ─────────────────────────────────────────────────
        col_clear, col_space = st.columns([1, 5])
        with col_clear:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if not llm_enabled:
            st.warning(
                "Set `GROQ_API_KEY` at the top of app.py to enable the chatbot."
            )
        else:
            # ── Step 1 : capture input FIRST, append to history immediately ──
            # This must happen BEFORE the display loop so the user message is
            # in session state before anything is rendered. If we render first
            # then append, the message disappears on the next rerun because
            # st.chat_input resets to None after every submission.
            user_question = st.chat_input(
                "Ask a question about this patient's prediction..."
            )

            if user_question:
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_question}
                )

            # ── Step 2 : render the full conversation from session state ─────
            # Every message — including the one just appended — is drawn here.
            # Because it comes from session state it survives any future rerun.
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # ── Step 3 : if the last message is from the user, get a reply ───
            # We check the last role instead of using `if user_question` so
            # this also handles edge cases where the stream was interrupted.
            if (st.session_state.chat_history
                    and st.session_state.chat_history[-1]["role"] == "user"):

                system_prompt = build_chat_system_prompt(
                    patient_info      = st.session_state.patient_context,
                    prediction_result = st.session_state.prediction_ctx,
                    prob_positive     = st.session_state.prob_ctx,
                    top_factors       = st.session_state.top_factors_ctx
                )

                # Full message list: system prompt + entire history
                messages_for_api = (
                    [{"role": "system", "content": system_prompt}]
                    + st.session_state.chat_history
                )

                with st.chat_message("assistant"):
                    try:
                        client = Groq(api_key=GROQ_API_KEY)

                        stream = client.chat.completions.create(
                            model      = "llama-3.3-70b-versatile",
                            messages   = messages_for_api,
                            max_tokens = 300,
                            stream     = True
                        )

                        # Stream tokens live — returns full text when done
                        full_response = st.write_stream(
                            chunk.choices[0].delta.content or ""
                            for chunk in stream
                            if chunk.choices[0].delta.content
                        )

                    except Exception as e:
                        full_response = f"Sorry, I could not generate a response: {e}"
                        st.error(full_response)

                # Append assistant reply — now both turns are in history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_response}
                )
