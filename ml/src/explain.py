import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, shap 

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

def load_rf_pipeline():
    all_models = joblib.load(MODELS_DIR / "all_models.joblib")
    return all_models["Random Forest Regression"]

def build_explainer(rf_pipeline, X_background: pd.DataFrame):
    preprocessor = rf_pipeline.named_steps["pre"]
    rf_model = rf_pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(rf_model)
    feature_names = list(preprocessor.get_feature_names_out())
    return explainer, feature_names

def _clean_name(raw_name: str) -> str:
    name = raw_name.split("__", 1)[-1]
    for prefix in ("location_", "house_style_", "kitchen_quality_", "central_air_"):
        if name.startswith(prefix):
            return f"{prefix.rstrip('_')}: {name[len(prefix): ]}"
    return name 

def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import ALL_INPUT_FEATURES, RAW_TO_FRIENDLY, engineer_features

    df = pd.read_csv(BASE_DIR / "data" / "train.csv").rename(columns=RAW_TO_FRIENDLY)
    df = engineer_features(df)
    X = df[ALL_INPUT_FEATURES]

    rf_pipeline = load_rf_pipeline()
    sample = X.sample(n=min(300, len(X)), random_state=42)
    explainer, feature_names = build_explainer(rf_pipeline, sample)

    preprocessor = rf_pipeline.named_steps["pre"]
    X_transformed = preprocessor.transform(sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    shap_values = explainer.shap_values(X_transformed)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = sorted(zip(feature_names, mean_abs_shap), key=lambda p: p[1], reverse=True)
    global_importance = [
        {"feature": _clean_name(n), "mean_abs_shap": round(float(v), 2)}
        for n, v in importance[:15]
    ]

    with open(MODELS_DIR / "feature_importance.json", "w") as f:
        json.dump({"global_importance": global_importance}, f, indent=2)

if __name__ == "__main__":
    main()
