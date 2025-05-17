import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def modeling_with_autolog(filepath):
    # Load dataset yang sudah diprocessing
    df = pd.read_csv(filepath)

    # Pisahkan fitur target (Churn)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi model
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

    # Latih model
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    print("Akurasi:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    input_file = "dataset_preprocessing/telco-customer-churn_preprocessing.csv"

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Telco_Customer_Churn_Model")

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Modelling_autolog"):
        trained_model = modeling_with_autolog(input_file)