import mlflow
import pandas as pd
import sqlite3

def load_data():
    conn = sqlite3.connect("database/students.db")
    query = """
    SELECT
        s.age,
        s.sex,
        s.address,
        f.Medu,
        f.Fedu,
        f.internet,
        a.studytime,
        a.failures,
        a.absences,
        a.subject
    FROM students s
    JOIN family_background f ON s.student_id = f.student_id
    JOIN academics a ON s.student_id = a.student_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def main():
    model = mlflow.sklearn.load_model("models:/Logistic_Regression_Baseline/Production")
    df = load_data()
    preds = model.predict(df)
    print("Sample predictions:", preds[:10])

if __name__ == "__main__":
    main()