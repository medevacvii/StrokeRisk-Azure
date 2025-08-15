import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from stroke_predictor_pkl import predict_stroke_risk

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "strokerisk_tune_ensemble_model.pkl"
    if not model_path.exists():
        st.error(f"Model file not found: {model_path.name}")
        st.stop()
    return joblib.load(model_path)

def preprocess_input(raw):
    processed = {
        'age': float((raw['age'] - 43.23) / 22.61),
        'avg_glucose_level': float((raw['avg_glucose_level'] - 106.15) / 45.28),
        'bmi': float((raw['bmi'] - 28.89) / 7.85),
        'hypertension': int(raw.get('hypertension', 0)),
        'heart_disease': int(raw.get('heart_disease', 0)),
        'gender_Male': int(raw.get('gender') == 'Male'),
        'gender_Other': int(raw.get('gender') == 'Other'),
        'ever_married_Yes': int(raw.get('ever_married') == 'Yes'),
        'work_type_Never_worked': int(raw.get('work_type') == 'Never_worked'),
        'work_type_Private': int(raw.get('work_type') == 'Private'),
        'work_type_Self-employed': int(raw.get('work_type') == 'Self-employed'),
        'work_type_children': int(raw.get('work_type') == 'children'),
        'Residence_type_Urban': int(raw.get('Residence_type') == 'Urban'),
        'smoking_status_formerly smoked': int(raw.get('smoking_status') == 'formerly smoked'),
        'smoking_status_never smoked': int(raw.get('smoking_status') == 'never smoked'),
        'smoking_status_smokes': int(raw.get('smoking_status') == 'smokes'),
        'age_group_19-30': int(19 <= raw['age'] <= 30),
        'age_group_31-45': int(31 <= raw['age'] <= 45),
        'age_group_46-60': int(46 <= raw['age'] <= 60),
        'age_group_61-75': int(61 <= raw['age'] <= 75),
        'age_group_76+': int(raw['age'] > 75),
    }
    cols = [
        'age','hypertension','heart_disease','avg_glucose_level','bmi',
        'gender_Male','gender_Other','ever_married_Yes',
        'work_type_Never_worked','work_type_Private','work_type_Self-employed','work_type_children',
        'Residence_type_Urban',
        'smoking_status_formerly smoked','smoking_status_never smoked','smoking_status_smokes',
        'age_group_19-30','age_group_31-45','age_group_46-60','age_group_61-75','age_group_76+'
    ]
    return pd.DataFrame([processed], columns=cols)

def patient_data_entry():
    st.set_page_config(page_title="Patient Data Entry", layout="wide")
    st.title("🩺 Stroke Risk Assessment System")
    st.session_state.setdefault('risk_result', None)
    st.session_state.setdefault('patient_data', None)
    st.session_state.setdefault('patient_id', None)

    pipe = load_model()

    with st.form("patient_form"):
        st.subheader("Patient Identification")
        patient_id = st.text_input("Patient ID/Name", placeholder="PT-001")

        st.subheader("Medical Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 50)
            hypertension = st.radio("Hypertension", ["No", "Yes"], index=0)
            heart_disease = st.radio("Heart Disease", ["No", "Yes"], index=0)
            avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", 50.0, 300.0, 100.0)
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        with col2:
            gender = st.radio("Gender", ["Male", "Female", "Other"], index=1)
            ever_married = st.radio("Ever Married", ["No", "Yes"], index=1)
            work_type = st.selectbox("Work Type", ["Private","Self-employed","Govt_job","children","Never_worked"], index=0)
            residence_type = st.radio("Residence Type", ["Urban","Rural"], index=0)
            smoking_status = st.selectbox("Smoking Status", ["never smoked","formerly smoked","smokes","Unknown"], index=0)

        submitted = st.form_submit_button("Assess Stroke Risk", type="primary")

    if submitted:
        st.session_state.patient_id = patient_id
        patient_data = {
            'age': age,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'gender': gender,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'smoking_status': smoking_status
        }
        st.session_state.patient_data = patient_data

        try:
            processed_df = preprocess_input(patient_data)
            probs = predict_stroke_risk(pipe, processed_df)  # <- no align function
            high = float(probs["High Risk"]); low = 1.0 - high
            risk_level = "High Risk" if high >= 0.5 else "Low Risk"
            st.session_state.risk_result = {
                'patient_id': patient_id,
                'risk_level': risk_level,
                'probability_percent': f"{high*100:.1f}%",
                'probabilities': [low, high],
                'probability_raw': high,
                'input_data': patient_data
            }
            st.success("Assessment completed!")
            st.balloons()
        except Exception as e:
            st.error(f"Assessment failed: {e}")
            st.session_state.risk_result = None

    if st.session_state.get('risk_result'):
        display_results(pipe)

def display_results(pipe):
    # ... keep your existing display code ...
    pass  # (omit for brevity; unchanged from your current version)

if __name__ == "__main__":
    patient_data_entry()