import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Configuracion del servidor de MLflow (1.4)
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

# Cargar el conjunto de datos desde el archivo CSV
try:
    iris = pd.read_csv('data/iris_dataset.csv')
except FileNotFoundError:
    print("Error: El archivo 'data/iris_dataset.csv' no fue encontrado.")
    exit(1)

# Dividir el DataFrame en caracteristicas (X) y etiquetas (y)
X = iris.drop('target', axis=1)
y = iris['target']

# Iniciar un experimento de MLflow
with mlflow.start_run():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, 'model.pkl')
    mlflow.sklearn.log_model(model, name="random-forest-model", serialization_format="cloudpickle")

    mlflow.log_param("n_estimators", 300)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Modelo entrenado y precision: {accuracy:.4f}")
    print("Experimento registrado con MLflow.")

    # --- Seccion de Reporte para CML (1.2) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusion')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    plt.savefig('confusion_matrix.png')
    print("Matriz de confusion guardada como 'confusion_matrix.png'")
    # --- Fin de la seccion de Reporte ---

    # Guardar el artefacto en MLflow remoto
    mlflow.log_artifact("confusion_matrix.png")