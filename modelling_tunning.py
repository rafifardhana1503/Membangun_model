import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os

def modeling_with_tuning(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Pisahkan fitur target (Churn)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menentukan hyperparameter grid
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, 15],
        "min_samples_split": [2, 4]
    }

    # Inisialisasi model & GridSearch
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        model, param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Prediksi & Evaluasi
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Hasil
    print("Akurasi:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return best_model, accuracy, report, grid_search.best_params_

if __name__ == "__main__":
    input_file = "dataset_preprocessing/telco-customer-churn_preprocessing.csv"

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Telco_Customer_Churn_Model_Tunning")

    with mlflow.start_run(run_name="Modelling_tunning_manuallog"):
        model, accuracy, report, best_params = modeling_with_tuning(input_file)

        # Log params
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        # Set tag untuk menandai tahap lifecycle model
        mlflow.set_tag("stage", "tunning")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        # Simpan model ke dalam MLflow dengan nama artifact rf_model
        mlflow.sklearn.log_model(model, artifact_path="rf_best_model")

        print("Proses tunning dan logged MLflow selesai")