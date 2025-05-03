import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# Public HTTPS URL for your S3‐hosted model (derived from s3://otc-only-model/…)
MODEL_URL = "https://otc-only-model.s3.amazonaws.com/otc_classifier_no_postpain.pkl"
#───────────────────────────────────────────────────────────────────────────────

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_artifacts():
    # 1) load preprocessor from local file
    pre = joblib.load('otc_preprocessor_no_postpain.pkl')

    # 2) download and load classifier from S3
    r = requests.get(MODEL_URL)
    r.raise_for_status()
    clf = joblib.load(BytesIO(r.content))

    # 3) load df just for populating dropdown options
    df = pd.read_csv('OTC-Data.csv', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Best OTC':'best_otc',
        'OTCSleep':'otc_sleep',
        'OTC Cause':'otc_cause',
        'OTC PainLocation':'otc_pain_location',
        'OTC PainTime':'otc_pain_time',
        'OTC CocomtSymptom':'otc_cocomt_symptom',
        'Gender':'gender',
        'Age':'age','Height':'height','Weight':'weight',
        'Ethnicity':'ethnicity','Race':'race'
    })
    # global grouping of rares (<5) into "Other"
    vc   = df['best_otc'].value_counts()
    rare = vc[vc < 5].index
    df['best_otc'] = df['best_otc'].apply(lambda x: 'Other' if x in rare else x)

    return pre, clf, df

preprocessor, model, df_full = load_artifacts()

st.title("🏥 OTC Knee Pain Recommender")

# ─── PATIENT PROFILE ───────────────────────────────────────────────────────────
st.header("Patient Profile")
age       = st.text_input("Age", value="", placeholder="e.g. 45")
gender    = st.selectbox("Gender", ["", *df_full['gender'].unique()])
race      = st.selectbox("Race", ["", *df_full['race'].unique()])
ethnicity = st.selectbox("Hispanic Origin/Ethnicity", ["", *df_full['ethnicity'].unique()])
weight    = st.text_input("Weight (lbs)", value="", placeholder="e.g. 150")
height    = st.text_input("Height (inches)", value="", placeholder="e.g. 65")
# ──────────────────────────────────────────────────────────────────────────────

# ─── PAIN LEVEL ────────────────────────────────────────────────────────────────
st.header("Pain Level")
pain_level = st.text_input(
    "Current pain level (1 = low, 10 = high)",
    value="",
    placeholder="Enter 1–10"
)
# ──────────────────────────────────────────────────────────────────────────────

# ─── PAIN CONTEXT ──────────────────────────────────────────────────────────────
st.header("Pain Context")
loc_opts = [
    "", "In the front of your knee", "All over the knee",
    "Close to the surface above or behind your knee",
    "Deeper inside your knee", "In multiple parts of your knee or leg",
    "None of the above"
]
pain_location = st.selectbox("Where do you feel your knee pain?", loc_opts)

time_opts = [
    "", "When moving or bending (better with rest)",
    "First thing in the morning", "More pain at night after activity",
    "During bad weather", "When stressed/anxious/tired",
    "When unwell", "None of the above"
]
pain_time = st.selectbox("When do you feel pain?", time_opts)

symp_opts = [
    "", "Dull pain", "Throbbing pain", "Sharp pain", "Swelling",
    "Stiffness", "Redness and warmth", "Instability or weakness",
    "Popping or crunching noises", "Limited range of motion",
    "Locking of the knee joint", "Inability to bear weight",
    "Fever", "Disabling pain", "Others", "None"
]
symptoms = st.multiselect("Accompanying symptoms", symp_opts)
# ──────────────────────────────────────────────────────────────────────────────

# ─── SLEEP & OTHER JOINT PAIN ──────────────────────────────────────────────────
st.header("Sleep & Other Joint Pain")
sleep_opts = ["", "Abnormal sleep pattern", "Pain at other joint(s)", "None of the above"]
sleep = st.selectbox("Do you experience any of these?", sleep_opts)
# ──────────────────────────────────────────────────────────────────────────────

# ─── CAUSE ─────────────────────────────────────────────────────────────────────
st.header("Likely Cause of Pain")
cause_opts = [
    "", "Overweight or obesity",
    "Injuries (ligaments/cartilage/bone fractures)",
    "Medical conditions (arthritis, gout, infections, tendonitis, bursitis)",
    "Aging (osteoarthritis)", "Repeated stress (overuse)",
    "Other conditions (patellofemoral syndrome, lupus, rheumatoid arthritis)",
    "None of the above", "Don’t know"
]
cause = st.selectbox("What caused your knee pain?", cause_opts)
# ──────────────────────────────────────────────────────────────────────────────

# ─── PREDICTION ────────────────────────────────────────────────────────────────
if st.button("Get OTC Recommendations"):
    # 1) Validate
    required = [age, gender, race, ethnicity, weight, height,
                pain_level, pain_location, pain_time, sleep, cause]
    if not all(required):
        st.error("Please fill in every field.")
    else:
        # 2) convert numeric
        try:
            age_v   = int(age)
            w_v     = float(weight)
            h_v     = float(height)
            pl_v    = int(pain_level)
        except ValueError:
            st.error("Age, weight, height must be numbers; pain level 1–10.")
            st.stop()

        # 3) build DataFrame
        input_df = pd.DataFrame([{
            'otc_prepain':    pl_v,
            'age':            age_v,
            'height':         h_v,
            'weight':         w_v,
            'gender':         gender,
            'race':           race,
            'ethnicity':      ethnicity,
            'otc_pain_location': pain_location,
            'otc_pain_time':     pain_time,
            'otc_cocomt_symptom': ",".join(symptoms) if symptoms else "",
            'otc_sleep':          sleep,
            'otc_cause':          cause
        }])

        # 4) predict
        Xp    = preprocessor.transform(input_df)
        probs = model.predict_proba(Xp)[0]
        classes = model.classes_
        top3 = probs.argsort()[-3:][::-1]

        st.subheader("Top 3 OTC Recommendations")
        for i in top3:
            st.write(f"- **{classes[i]}**: {probs[i]:.1%} confidence")
