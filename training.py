from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import mlflow
import mlflow.sklearn

# Importing metrics for evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# This is a dummy dataset. Replace this with your own data loading logic.
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import datetime

# data = load_iris()

import os
import pandas as pd

def preprocess_data(df):
    # Your preprocessing steps, for example:
    one_hot_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
    one_hot_encoded.rename(columns={'Sex_M': 'Sex', 'ExerciseAngina_Y': 'ExerciseAngina'}, inplace=True)
    return one_hot_encoded

# Paths to the files
original_data_path = "heart.csv"
preprocessed_data_path = "preprocessed_heart.csv"

# Check if preprocessed data exists
if os.path.exists(preprocessed_data_path):
    df = pd.read_csv(preprocessed_data_path)
else:
    # Load the original dataset
    df_original = pd.read_csv(original_data_path)
    # Preprocess the data
    df = preprocess_data(df_original)
    # Save the preprocessed data for future use
    df.to_csv(preprocessed_data_path, index=False)



X = df.drop('HeartDisease', axis=1).values
y = df['HeartDisease'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def train_model(model_type, n_estimators=100, max_depth=None, C=1.0):
    with mlflow.start_run():
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        unique_run_name = f"{model_type}_{current_time}"
        mlflow.set_tag("mlflow.runName", unique_run_name)
        if model_type == "RandomForest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            mlflow.log_params({
                "model_type": model_type,
                "n_estimators": n_estimators,
                "max_depth": max_depth
            })
        elif model_type == "LogisticRegression":
            model = LogisticRegression(max_iter=1000, C=C)
            mlflow.log_params({
                "model_type": model_type,
                "C": C
            })
        elif model_type == "SVM":
            model = svm.SVC(C=C)
            mlflow.log_params({
                "model_type": model_type,
                "C": C
            })

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        conf_matrix = confusion_matrix(y_test, predictions)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix as an artifact (if you want to view it later)
        with open("confusion_matrix.txt", "w") as f:
            f.write(str(conf_matrix))
        mlflow.log_artifact("confusion_matrix.txt")

        print(f"{model_type} Model Trained with Accuracy: {accuracy}")
