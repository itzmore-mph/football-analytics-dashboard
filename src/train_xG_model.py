import os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define file paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "processed_shots.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_xg_model.pkl")

def train_xg_model():
    """Trains an xG model using XGBoost and saves it."""
    if not os.path.exists(DATA_PATH):
        print(f"Missing processed data file: {DATA_PATH}")
        return

    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Ensure required columns exist
    required_columns = ["shot_distance", "shot_angle", "goal_scored"]
    if not all(col in df.columns for col in required_columns):
        print(f"Data is missing required columns: {required_columns}")
        return

    X = df[["shot_distance", "shot_angle"]]
    y = df["goal_scored"]

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.2f}")

    # Save model
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_xg_model()
