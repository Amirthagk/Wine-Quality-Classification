import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    # Load the wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MLflow Setup
    mlflow.set_experiment("Wine_Quality_Classification")
    
    with mlflow.start_run():
        # Define model parameters
        n_estimators = 100
        max_depth = 5
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions and evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Save model locally for FastAPI
        joblib.dump(model, "model.joblib")
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print("Model saved as model.joblib")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=wine.target_names))

if __name__ == "__main__":
    train_model()
