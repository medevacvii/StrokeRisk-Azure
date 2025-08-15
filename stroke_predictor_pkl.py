# stroke_predictor_pkl.py

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

# ---------------- internal helpers ----------------
def _unwrap_estimator(est):
    """Drill down through nested Pipelines/Calibrators to the actual estimator."""
    seen = set()
    while True:
        if id(est) in seen:
            break
        seen.add(id(est))

        # CalibratedClassifierCV or similar
        if hasattr(est, "base_estimator"):
            est = est.base_estimator
            continue

        # Nested pipeline as final step
        if isinstance(est, Pipeline):
            for key in ("model", "classifier", "clf", "final_estimator", "ensemble"):
                if key in est.named_steps:
                    est = est.named_steps[key]
                    break
            else:
                est = list(est.named_steps.values())[-1]
            continue
        break
    return est

def _expected_feature_names(pipe):
    """Names after preprocessor; may be None if none available."""
    prep = pipe.named_steps.get("prep") if hasattr(pipe, "named_steps") else None
    if prep is None:
        return None
    if hasattr(prep, "get_feature_names_out"):
        try:
            return list(prep.get_feature_names_out())
        except Exception:
            pass
    if hasattr(prep, "feature_names_in_"):
        return list(prep.feature_names_in_)
    return None

def feature_importances_from_pipeline(pipe):
    """Return DataFrame(feature, importance) for tree models or weighted VotingClassifier."""
    feat_names = _expected_feature_names(pipe)

    if hasattr(pipe, "named_steps"):
        final = pipe.named_steps.get("model") or pipe.named_steps.get("ensemble")
    else:
        final = None
    final = final if final is not None else pipe

    est = _unwrap_estimator(final)

    # Single tree-like model
    if hasattr(est, "feature_importances_"):
        imps = np.asarray(est.feature_importances_, dtype=float)
        if feat_names is None:
            feat_names = [f"f{i}" for i in range(len(imps))]
        return (pd.DataFrame({"feature": feat_names, "importance": imps})
                .sort_values("importance", ascending=False).reset_index(drop=True))

    # Voting ensemble of trees
    if isinstance(final, VotingClassifier) and hasattr(final, "estimators_"):
        weights = final.weights or [1.0] * len(final.estimators_)
        total, wsum = None, 0.0
        for w, (_, base) in zip(weights, final.estimators_):
            base = _unwrap_estimator(base)
            if hasattr(base, "feature_importances_"):
                vec = np.asarray(base.feature_importances_, dtype=float)
                total = vec * w if total is None else total + vec * w
                wsum += w
        if total is None or wsum == 0:
            raise AttributeError("None of the base estimators expose feature_importances_.")
        imps = total / wsum
        if feat_names is None:
            feat_names = [f"f{i}" for i in range(len(imps))]
        return (pd.DataFrame({"feature": feat_names, "importance": imps})
                .sort_values("importance", ascending=False).reset_index(drop=True))

    # No native importances
    raise AttributeError("Model exposes no feature_importances_.")

# ---------------- compatibility wrappers used by pages ----------------
def get_feature_importance(model_or_pipeline):
    """Back-compat: used by 02_ and 03_ pages."""
    return feature_importances_from_pipeline(model_or_pipeline)

def _expected_input_names(model_or_pipeline):
    """Columns the pipeline expects *as input* (before transformation)."""
    if hasattr(model_or_pipeline, "named_steps"):
        prep = model_or_pipeline.named_steps.get("prep")
        if prep is not None:
            # Prefer input names (this is what ColumnTransformer selects on)
            if hasattr(prep, "feature_names_in_"):
                return list(prep.feature_names_in_)
            # Fallback: pull explicit column lists from transformers
            try:
                cols = []
                for name, trans, colsel in getattr(prep, "transformers", []):
                    if isinstance(colsel, (list, tuple)):
                        cols.extend(colsel)
                return list(cols) if cols else None
            except Exception:
                pass
    return None

def predict_stroke_risk(model_or_pipeline, input_df):
    """Return {"Low Risk", "High Risk"}; auto-align to pipeline input schema."""
    # 1) Coerce to 1-row DataFrame
    df = pd.DataFrame([input_df]) if isinstance(input_df, dict) else input_df.copy()

    # 2) Align to INPUT schema, not the transformed names
    expected_in = _expected_input_names(model_or_pipeline)
    if expected_in is not None:
        df = df.reindex(columns=expected_in, fill_value=0.0)

    # 3) Predict
    proba = float(model_or_pipeline.predict_proba(df)[0, 1])
    return {"Low Risk": 1.0 - proba, "High Risk": proba}

def generate_what_if_scenario(model_or_pipeline, base_row_df, feature, values):
    """Simple What-If: vary one feature and return risks as a DataFrame."""
    out = []
    for v in values:
        row = base_row_df.copy()
        row.iloc[0, row.columns.get_loc(feature)] = v
        proba = float(model_or_pipeline.predict_proba(row)[0, 1])
        out.append({"feature": feature, "value": v, "High Risk": proba, "Low Risk": 1.0 - proba})
    return pd.DataFrame(out)
