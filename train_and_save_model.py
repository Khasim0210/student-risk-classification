import sqlite3
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# -----------------------------
# Config
# -----------------------------
DB_PATH = "database/students.db"
MODEL_OUT_PATH = "model.joblib"

# These MUST match your Streamlit app REQUIRED_COLUMNS order/meaning
FEATURE_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "internet", "studytime", "failures", "absences", "subject"
]
TARGET_COLUMN = "at_risk"


def load_training_data_from_sqlite(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        s.school,
        s.sex,
        s.age,
        s.address,
        f.famsize,
        f.Pstatus,
        f.Medu,
        f.Fedu,
        f.internet,
        a.studytime,
        a.failures,
        a.absences,
        a.subject,
        a.at_risk
    FROM students s
    JOIN family_background f ON s.student_id = f.student_id
    JOIN academics a ON s.student_id = a.student_id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model),
    ])

    return pipeline


def main():
    # 1) Load
    df = load_training_data_from_sqlite(DB_PATH)

    # 2) Keep only expected columns (safe)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"DB query missing columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    # 3) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Build + Train
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    # 5) Evaluate (optional but useful)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("Accuracy:", acc)
    print("F1:", f1)
    print("\nClassification report:\n", classification_report(y_test, preds))

    # 6) Save to repo root as model.joblib
    joblib.dump(pipeline, MODEL_OUT_PATH)
    print(f"\n✅ Saved model to: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
