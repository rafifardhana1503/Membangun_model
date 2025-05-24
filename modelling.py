import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def modeling_with_autolog(X_train_path, X_test_path, y_train_path, y_test_path):
    # Load dataset yang sudah diprocessing
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

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
    # Path file hasil preprocessing dan split
    X_train_path = "dataset_preprocessing/X_train.csv"
    X_test_path = "dataset_preprocessing/X_test.csv"
    y_train_path = "dataset_preprocessing/y_train.csv"
    y_test_path = "dataset_preprocessing/y_test.csv"

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Telco_Customer_Churn_Model")

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Modelling_autolog"):
        trained_model = modeling_with_autolog(X_train_path, X_test_path, y_train_path, y_test_path)