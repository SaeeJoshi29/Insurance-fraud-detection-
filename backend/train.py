import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from app.config import DATA_PATH, MODELS_DIR, TARGET, ID_COLUMNS, RANDOM_STATE

MODEL_BUILDERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=3000, class_weight=None, C=1.0, random_state=RANDOM_STATE),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=15, weights="distance", metric="minkowski"),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2, class_weight=None, n_jobs=-1, random_state=RANDOM_STATE),
    "SVM": lambda: SVC(C=1.5, kernel="rbf", probability=True, class_weight=None, random_state=RANDOM_STATE),
}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # Dataset uses '.' in some categorical fields for missing values.
    df = df.replace({".": np.nan, "": np.nan, "?": np.nan})
    # Age=0 is an unknown/invalid age in this dataset; let the numeric imputer handle it.
    if "Age" in df.columns:
        df.loc[df["Age"] == 0, "Age"] = np.nan
    return df


def make_schema(df: pd.DataFrame, model_features):
    fields = []
    groups = {
        "Claim & Timing": ["Month", "WeekOfMonth", "DayOfWeek", "DayOfWeekClaimed", "MonthClaimed", "WeekOfMonthClaimed", "Year"],
        "Policyholder": ["Sex", "MaritalStatus", "Age", "AgeOfPolicyHolder"],
        "Vehicle": ["Make", "VehicleCategory", "VehiclePrice", "AgeOfVehicle", "NumberOfCars"],
        "Accident & Claim": ["AccidentArea", "Fault", "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims", "PoliceReportFiled", "WitnessPresent", "NumberOfSuppliments", "AddressChange_Claim"],
        "Policy & Agent": ["PolicyType", "Deductible", "DriverRating", "AgentType", "RepNumber", "BasePolicy"],
    }
    reverse = {f: g for g, fs in groups.items() for f in fs}
    for col in model_features:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        if numeric:
            values = s.dropna()
            field = {
                "name": col,
                "type": "number",
                "group": reverse.get(col, "Other"),
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
                "default": float(values.median()) if len(values) else 0,
            }
        else:
            values = sorted(str(v) for v in s.dropna().unique())
            field = {
                "name": col,
                "type": "select",
                "group": reverse.get(col, "Other"),
                "options": values,
                "default": values[0] if values else "",
            }
        fields.append(field)
    return {"target": TARGET, "excluded_columns": ID_COLUMNS, "model_features": list(model_features), "fields": fields}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Place fraud_oracle.csv at: {DATA_PATH}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_raw(pd.read_csv(DATA_PATH))
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column {TARGET}")

    y = df[TARGET].astype(int)
    # PolicyNumber is an identifier and must not become a model feature.
    model_features = [c for c in df.columns if c not in [TARGET, *ID_COLUMNS]]
    X = df[model_features].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    numeric_cols = [c for c in model_features if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in model_features if c not in numeric_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols),
    ])

    metrics = {}
    for name, builder in MODEL_BUILDERS.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            # Critical: SMOTE is inside the imbalanced-learn pipeline, so it only sees training folds/data.
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("model", builder()),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, pred).tolist()
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "confusion_matrix": cm,
        }
        joblib.dump(pipeline, MODELS_DIR / {
            "Logistic Regression": "logistic_regression.joblib",
            "KNN": "knn.joblib",
            "Random Forest": "random_forest.joblib",
            "SVM": "svm.joblib",
        }[name])
        print(name, metrics[name])

    schema = make_schema(df, model_features)
    schema.update({
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "dataset_rows": int(len(df)),
        "fraud_count": int(y.sum()),
        "genuine_count": int((y == 0).sum()),
    })
    (MODELS_DIR / "metrics.json").write_text(json.dumps({
        "dataset": {"rows": len(df), "fraud": int(y.sum()), "genuine": int((y == 0).sum())},
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "models": metrics,
    }, indent=2))
    (MODELS_DIR / "schema.json").write_text(json.dumps(schema, indent=2))
    print("Training complete. Artifacts saved to", MODELS_DIR)

if __name__ == "__main__":
    main()
