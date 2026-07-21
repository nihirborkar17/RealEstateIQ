import json, time, warnings
from pathlib import Path

import joblib
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer ,TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from features import (ALL_INPUT_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_TO_FRIENDLY, TARGET, engineer_features)

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=RAW_TO_FRIENDLY)
    df = engineer_features(df)
    df = df[ALL_INPUT_FEATURES + [TARGET]]
    df = df.dropna(subset=[TARGET])
    return df

def build_preprocessor(poly: bool=False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if poly:
        numeric_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    numeric_steps.append(("scale", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat",  categorical_pipe, CATEGORICAL_FEATURES),
    ])

def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mse": round(float(mean_squared_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }

def train_all_models(X_train, X_test, y_train, y_test):
    results, fitted = {}, {}

    model_specs = [
        ("Simple Linear Regression", None, None),
        ("Multiple Linear Regression", 
         Pipeline([("pre", build_preprocessor()), ("model", LinearRegression())]),
         None),
        ("Polynomial Regression", 
         Pipeline([("pre", build_preprocessor(poly=True)), ("model", LinearRegression())]), 
         None),
        ("Support Vector Regression", 
         Pipeline([("pre", build_preprocessor()), ("model", TransformedTargetRegressor(regressor=SVR(kernel='rbf'), transformer=StandardScaler()))]),
         {"model__regressor__C": [1, 10, 100, 1000], "model__regressor__epsilon": [0.01, 0.1]}),
        ("Decision Tree Regression",
         Pipeline([("pre", build_preprocessor()),("model", DecisionTreeRegressor(random_state=RANDOM_STATE))]),
         {"model__max_depth":[5, 10, 15, None], "model__min_samples_leaf": [1, 5, 10]}),
        ("Random Forest Regression", 
         Pipeline([("pre", build_preprocessor()), ("model", RandomForestRegressor(random_state=RANDOM_STATE))]),
         {"model__n_estimators": [100, 300], "model__max_depth": [10,20,None]}),
    ]

    for name, pipe, param_grid in model_specs:
        print(f"Training {name}...")
        start = time.time()

        if name == "Simple Linear Regression":
            X_train_use, X_test_use = X_train[["living_area_sqft"]], X_test[["living_area_sqft"]]
            simple_pre = ColumnTransformer([
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")), 
                                ("scale", StandardScaler())]), ["living_area_sqft"])
            ])
            pipe = Pipeline([("pre", simple_pre), ("model", LinearRegression())])
            pipe.fit(X_train_use, y_train)
            preds = pipe.predict(X_test_use)
        elif param_grid:
            search = GridSearchCV(pipe, param_grid, cv=3, scoring="r2", n_jobs=-1)
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
            preds = pipe.predict(X_test)
        else:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
        
        metrics = evaluate(y_test, preds)
        metrics["training_time_sec"] = round(time.time() - start, 2)
        results[name], fitted[name] = metrics, pipe 
        print(f"  -> R2={metrics['r2']}   RMSE={metrics['rmse']}")

    return results, fitted

def main():
    df = load_data()
    X, y = df[ALL_INPUT_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results, fitted = train_all_models(X_train, X_test, y_train, y_test)
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = fitted[best_name]

    joblib.dump(best_model, MODELS_DIR/"best_model.joblib")
    joblib.dump(fitted, MODELS_DIR/"all_models.joblib")

    metrics_out = {
        "best_model": best_name,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "dataset_rows": len(df),
        "models": results,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

if __name__ == "__main__":
    main()