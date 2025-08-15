# pages/03_Risk_Assessment_Results.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import os, joblib, pickle, hashlib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

# only need this wrapper; it aligns to the pipeline's INPUT schema internally
from stroke_predictor_pkl import predict_stroke_risk

# ---------------- Cache helper ----------------
try:
    cache_resource = st.cache_resource
except Exception:
    cache_resource = st.cache
    
def _sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1<<20)
            if not b: break
            h.update(b)
    return h.hexdigest()

def _is_lfs_pointer(path):
    try:
        with open(path, "rb") as f:
            head = f.read(200)
        return b"git-lfs" in head or head.startswith(b"version https://git-lfs")
    except Exception:
        return False

def _try_load(path):
    # joblib first
    try:
        return joblib.load(path), None
    except Exception as e1:
        # pickle fallback
        try:
            with open(path, "rb") as f:
                return pickle.load(f), f"pickle fallback after joblib: {e1}"
        except Exception as e2:
            return None, f"{e1} | {e2}"

# ---------------- Model loader (no st.stop; shows diagnostics) ----------------
@cache_resource
def load_models():
    model_dir = Path(__file__).resolve().parent.parent
    items = [
        ("Random Forest", "strokerisk_model_rf.pkl"),
        ("XGBoost", "strokerisk_model_xgboost.pkl"),
        ("Extra Trees", "strokerisk_model_et.pkl"),
        ("Ensemble", "strokerisk_tune_ensemble_model.pkl"),
    ]
    # alternate filenames to try if the .pkl is broken/missing
    alternates = {
        "strokerisk_model_et.pkl": "strokerisk_model_et.joblib",
    }

    models, diag = {}, {"files": [] , "missing": [], "failed": []}

    for label, fname in items:
        tried = []
        primary = model_dir / fname
        cand = [primary]
        alt = alternates.get(fname)
        if alt:
            cand.append(model_dir / alt)

        loaded = False
        for c in cand:
            if not c.exists():
                continue
            sz = c.stat().st_size
            ptr = _is_lfs_pointer(c)
            sha = _sha256(c) if sz < 50_000_000 else "(skipped)"
            diag["files"].append({"label": label, "file": c.name, "bytes": sz, "sha256": sha, "lfs_pointer": ptr})
            model, note = _try_load(c)
            tried.append((c.name, sz, note))
            if model is not None:
                if note:
                    st.warning(f"{label}: {note}")
                models[label] = model
                loaded = True
                break

        if not loaded:
            if not any((model_dir / n).exists() for n in [fname, alternates.get(fname, "")]):
                diag["missing"].append((label, fname))
            else:
                diag["failed"].append((label, fname, tried))

    return models, diag, model_dir

# ---------------- Preprocess: match your training features ----------------
def preprocess_input(input_data):
    processed = {
        'age': float((input_data['age'] - 43.23) / 22.61),
        'avg_glucose_level': float((input_data['avg_glucose_level'] - 106.15) / 45.28),
        'bmi': float((input_data['bmi'] - 28.89) / 7.85),
        'hypertension': int(input_data.get('hypertension', 0)),
        'heart_disease': int(input_data.get('heart_disease', 0)),
        'gender_Male': int(input_data.get('gender') == 'Male'),
        'gender_Other': int(input_data.get('gender') == 'Other'),
        'ever_married_Yes': int(input_data.get('ever_married') == 'Yes'),
        'work_type_Never_worked': int(input_data.get('work_type') == 'Never_worked'),
        'work_type_Private': int(input_data.get('work_type') == 'Private'),
        'work_type_Self-employed': int(input_data.get('work_type') == 'Self-employed'),
        'work_type_children': int(input_data.get('work_type') == 'children'),
        'Residence_type_Urban': int(input_data.get('Residence_type') == 'Urban'),
        'smoking_status_formerly smoked': int(input_data.get('smoking_status') == 'formerly smoked'),
        'smoking_status_never smoked': int(input_data.get('smoking_status') == 'never smoked'),
        'smoking_status_smokes': int(input_data.get('smoking_status') == 'smokes'),
        'age_group_19-30': int(19 <= input_data['age'] <= 30),
        'age_group_31-45': int(31 <= input_data['age'] <= 45),
        'age_group_46-60': int(46 <= input_data['age'] <= 60),
        'age_group_61-75': int(61 <= input_data['age'] <= 75),
        'age_group_76+': int(input_data['age'] > 75),
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

# ---------------- Importance helpers (robust + local fallback) ----------------
def _final_step(pipe):
    if hasattr(pipe, "named_steps"):
        for key in ("model", "ensemble", "classifier", "clf", "final_estimator"):
            if key in pipe.named_steps:
                return pipe.named_steps[key]
        return list(pipe.named_steps.values())[-1]
    return pipe

def _unwrap(est):
    return est.base_estimator if hasattr(est, "base_estimator") else est

def _expected_input_names(pipe):
    if hasattr(pipe, "named_steps"):
        prep = pipe.named_steps.get("prep")
        if prep is not None and hasattr(prep, "feature_names_in_"):
            return list(prep.feature_names_in_)
    return None

def _display_feature_names(pipe, fallback_cols):
    names = _expected_input_names(pipe)
    return names if names is not None else list(fallback_cols)

def tree_or_ensemble_importances(pipe, display_names):
    final = _final_step(pipe)
    est = _unwrap(final)

    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        return pd.DataFrame({"feature": display_names[:len(imp)], "importance": imp})

    if isinstance(final, VotingClassifier):
        if hasattr(final, "estimators_"):
            bases = [ _unwrap(e) for e in final.estimators_ ]
            weights = final.weights or [1.0] * len(bases)
        else:
            bases = [ _unwrap(e) for _, e in getattr(final, "estimators", []) ]
            weights = final.weights or [1.0] * len(bases)

        total, wsum = None, 0.0
        for w, base in zip(weights, bases):
            if hasattr(base, "feature_importances_"):
                vec = np.asarray(base.feature_importances_, dtype=float)
                total = vec * w if total is None else total + vec * w
                wsum += w
        if total is not None and wsum > 0:
            imp = total / wsum
            return pd.DataFrame({"feature": display_names[:len(imp)], "importance": imp})

    return None

def local_sensitivity_importance(pipe, base_row_df, input_cols):
    X0 = base_row_df.reindex(columns=input_cols, fill_value=0.0)
    base_p = float(pipe.predict_proba(X0)[0, 1])

    names, deltas = list(X0.columns), []
    for col in names:
        X1 = X0.copy()
        v = X1.iloc[0, X1.columns.get_loc(col)]
        if isinstance(v, (int, np.integer, float, np.floating)):
            step = 1.0 if abs(float(v)) <= 5 else max(0.05 * abs(float(v)), 1.0)
            X1.iloc[0, X1.columns.get_loc(col)] = float(v) + step
        else:
            X1.iloc[0, X1.columns.get_loc(col)] = 1
        try:
            p1 = float(pipe.predict_proba(X1)[0, 1])
        except Exception:
            p1 = base_p
        deltas.append(abs(p1 - base_p))

    imp = np.asarray(deltas, dtype=float)
    s = imp.sum()
    if s <= 0:
        imp = np.ones_like(imp)
    imp = imp / imp.sum()
    return pd.DataFrame({"feature": names, "importance": imp})

def compute_probs_and_importances(pipe, processed):
    # 1) probability
    probs = predict_stroke_risk(pipe, processed)
    proba = float(probs["High Risk"])

    # 2) names
    display_names = _display_feature_names(pipe, processed.columns)
    input_cols = _expected_input_names(pipe) or list(processed.columns)

    # 3) native importances
    imp_df = tree_or_ensemble_importances(pipe, display_names)

    # 4) fallback to local sensitivity
    if imp_df is None:
        imp_df = local_sensitivity_importance(pipe, processed, input_cols)

    imp_df["importance"] = pd.to_numeric(imp_df["importance"], errors="coerce").fillna(0.0)
    return proba, imp_df

# ---------------- Page ----------------
def risk_analysis_engine():
    st.set_page_config(layout="wide")
    st.title("🔬 Stroke Risk Analysis Engine")

    # Require patient data first
    if "patient_data" not in st.session_state:
        st.warning("No patient data found. Complete the assessment first.")
        if st.button("Go to Assessment"):
            st.switch_page("pages/02_Patient_Data_Entry.py")
        return

    # Load models (no stop)
    models, diag, model_dir = load_models()
    with st.expander("Diagnostics", expanded=False):
        st.write("Model dir:", str(model_dir))
        st.write("Loaded:", list(models.keys()))
        st.write(diag)

    if not models:
        st.error("No models loaded. Ensure .pkl files are in repo root, then restart.")
        return

    input_data = st.session_state["patient_data"]
    processed = preprocess_input(input_data)

    # Predictions + importances
    results = {}
    for name, pipe in models.items():
        try:
            proba, imp_df = compute_probs_and_importances(pipe, processed)
            results[name] = {
                "Low Risk": 1.0 - proba,
                "High Risk": proba,
                "ImportanceDF": imp_df
            }
        except Exception as e:
            st.error(f"{name} model failed: {e}")

    if not results:
        st.error("All models failed to predict. Check model compatibility.")
        return

    # -------- Summary --------
    with st.expander("📊 Risk Summary", expanded=True):
        avg_risk = float(np.mean([v["High Risk"] for v in results.values()])) * 100.0
        risk_color = "#FF4B4B" if avg_risk > 30 else ("#F9D423" if avg_risk > 15 else "#09AB3B")
        st.markdown(
            f"""<div style="background-color:{risk_color};padding:10px;border-radius:5px;color:white;text-align:center">
            <h2 style="color:white;">Average Stroke Risk: {avg_risk:.1f}%</h2></div>""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Risk Probability")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            for model_name, v in results.items():
                ax1.bar(model_name, v["High Risk"])
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("Probability")
            plt.xticks(rotation=45)
            st.pyplot(fig1)

        with col2:
            st.subheader("Model Agreement")
            df_probs = pd.DataFrame(
                {k: {"Low Risk": v["Low Risk"], "High Risk": v["High Risk"]} for k, v in results.items()}
            ).T
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            df_probs[["Low Risk", "High Risk"]].plot(kind="bar", stacked=True, ax=ax2)
            ax2.set_ylabel("Probability")
            st.pyplot(fig2)

    # -------- Feature Analysis --------
    with st.expander("🔍 Feature Analysis", expanded=False):
        tabs = st.tabs(list(results.keys()))
        for (name, v), tab in zip(results.items(), tabs):
            with tab:
                imp = v["ImportanceDF"].copy()
                imp["importance"] = pd.to_numeric(imp["importance"], errors="coerce").fillna(0.0)
                st.subheader(f"{name} Key Drivers")
                fig, ax = plt.subplots(figsize=(8, 6))
                imp.head(10).iloc[::-1].plot.barh(x="feature", y="importance", ax=ax)
                ax.set_title(f"Top 10 Predictive Features ({name})")
                st.pyplot(fig)
                st.dataframe(imp.reset_index(drop=True))

    # -------- Clinical Insights --------
    with st.expander("💡 Clinical Insights & Recommendations", expanded=False):
        risk_factors, protective = [], []
        if input_data.get('age', 0) > 60: risk_factors.append(f"Age ({input_data['age']})")
        if input_data.get('hypertension'): risk_factors.append("Hypertension")
        if input_data.get('avg_glucose_level', 0) > 140: risk_factors.append(f"High glucose ({input_data['avg_glucose_level']:.1f})")
        if input_data.get('smoking_status') in ['formerly smoked', 'smokes']: risk_factors.append("Smoking history")
        if input_data.get('age', 0) < 40: protective.append("Younger age")
        if not input_data.get('hypertension'): protective.append("No hypertension")
        if input_data.get('smoking_status') == 'never smoked': protective.append("Never smoked")

        avg_risk = float(np.mean([v["High Risk"] for v in results.values()])) * 100.0
        st.write(f"- **Consensus Risk**: {avg_risk:.1f}%")
        st.write(f"- **Key Risk Factors**: {', '.join(risk_factors) or 'None identified'}")
        st.write(f"- **Protective Factors**: {', '.join(protective) or 'None identified'}")
        st.subheader("Recommendations")
        if avg_risk > 30:
            st.warning("🚨 High Risk: consult a specialist; lifestyle interventions; monitor BP.")
        elif avg_risk > 15:
            st.info("⚠️ Moderate Risk: annual screening; maintain diet/exercise.")
        else:
            st.success("✅ Low Risk: continue preventive measures; routine check-ups.")

# run page
risk_analysis_engine()