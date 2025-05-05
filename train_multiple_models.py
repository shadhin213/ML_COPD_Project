import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import sys

def train_multiple_models():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Check if data file exists
    data_path = 'data/COPD_Data.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please make sure the COPD_Data.csv file is in the data directory.")
        sys.exit(1)

    try:
        # Load the data
        print("Loading data...")
        df = pd.read_csv(data_path)

        # Prepare features and target
        print("Preparing features...")
        # Drop patient name, uid, and target columns
        X = df.drop(['class', 'uid', 'patient_name'], axis=1)
        y = df['class']

        # Convert categorical variables
        X['sex'] = X['sex'].map({'Male': 1, 'Female': 0})
        X['smoke'] = X['smoke'].map({'Yes': 1, 'No': 0})
        
        # Encode location
        location_encoder = LabelEncoder()
        X['location'] = location_encoder.fit_transform(X['location'])

        # Scale numerical features
        print("Scaling features...")
        scaler = StandardScaler()
        numerical_cols = ['age', 'bmi', 'rs10007052', 'rs9296092']
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        # Split the data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 1. Train Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

        # 2. Train XGBoost
        print("\nTraining XGBoost...")
        xgb_model = XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")

        # Save the best model
        print("\nSaving models and preprocessing objects...")
        if rf_accuracy >= xgb_accuracy:
            print("Saving Random Forest as the best model...")
            with open('models/model_1.pkl', 'wb') as file:
                pickle.dump(rf_model, file)
        else:
            print("Saving XGBoost as the best model...")
            with open('models/model_1.pkl', 'wb') as file:
                pickle.dump(xgb_model, file)

        # Save the scaler and location encoder
        with open('models/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)
        with open('models/location_encoder.pkl', 'wb') as file:
            pickle.dump(location_encoder, file)

        # Print detailed classification reports
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_pred))
        print("\nXGBoost Classification Report:")
        print(classification_report(y_test, xgb_pred))

    except Exception as e:
        print(f"Error during model training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    train_multiple_models() 