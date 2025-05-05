import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import sys

def train_and_save_model():
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

        # Train the model
        print("Training model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the model
        print("Saving model...")
        with open('models/model_1.pkl', 'wb') as file:
            pickle.dump(model, file)

        # Save the scaler
        print("Saving scaler...")
        with open('models/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)
            
        # Save the location encoder
        print("Saving location encoder...")
        with open('models/location_encoder.pkl', 'wb') as file:
            pickle.dump(location_encoder, file)

        print("\nModel trained and saved successfully!")
        print(f"Training accuracy: {model.score(X_train, y_train):.2f}")
        print(f"Testing accuracy: {model.score(X_test, y_test):.2f}")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    train_and_save_model() 