from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import os
import sys
import io
from datetime import datetime

app = Flask(__name__)

def load_model_and_scaler():
    model_path = 'models/model_1.pkl'
    scaler_path = 'models/scaler.pkl'
    location_encoder_path = 'models/location_encoder.pkl'
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, location_encoder_path]):
        print("Error: Model files not found!")
        print("Please run 'python train_model.py' first to train and save the model.")
        sys.exit(1)
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        with open(location_encoder_path, 'rb') as file:
            location_encoder = pickle.load(file)
        return model, scaler, location_encoder
    except Exception as e:
        print(f"Error loading model files: {str(e)}")
        sys.exit(1)

# Load the model, scaler, and location encoder
model, scaler, location_encoder = load_model_and_scaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        patient_name = request.form['patient_name']
        data = {
            'sex': request.form['sex'],
            'age': float(request.form['age']),
            'bmi': float(request.form['bmi']),
            'smoke': request.form['smoke'],
            'location': request.form['location'],
            'rs10007052': float(request.form['rs10007052']),
            'rs9296092': float(request.form['rs9296092'])
        }
        
        # Convert categorical variables
        data['sex'] = 1 if data['sex'] == 'Male' else 0
        data['smoke'] = 1 if data['smoke'] == 'Yes' else 0
        
        # Handle location encoding with fallback
        try:
            data['location'] = location_encoder.transform([data['location']])[0]
        except ValueError:
            data['location'] = location_encoder.transform(['Urban'])[0]
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Scale numerical features
        numerical_cols = ['age', 'bmi', 'rs10007052', 'rs9296092']
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)
        
        result = {
            'patient_name': patient_name,
            'prediction': 'COPD Detected' if prediction[0] == 1 else 'No COPD Detected',
            'probability': f"{probability[0][1]*100:.2f}%",
            'download_data': {
                'Patient Name': patient_name,
                'Sex': 'Male' if data['sex'] == 1 else 'Female',
                'Age': request.form['age'],
                'BMI': request.form['bmi'],
                'Smoking Status': request.form['smoke'],
                'Location': request.form['location'],
                'RS10007052': request.form['rs10007052'],
                'RS9296092': request.form['rs9296092'],
                'Prediction': 'COPD Detected' if prediction[0] == 1 else 'No COPD Detected',
                'Probability': f"{probability[0][1]*100:.2f}%",
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_result', methods=['POST'])
def download_result():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Create a string buffer
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        # Generate filename with patient name and timestamp
        filename = f"COPD_Prediction_{data['Patient Name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return send_file(
            io.BytesIO(buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)