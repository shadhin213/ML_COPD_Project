import pandas as pd
import numpy as np
import os
from faker import Faker

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()  # Using English names

# Number of samples
n_samples = 1000

# Define location options
locations = ['Urban', 'Rural', 'Suburban', 'Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 
            'Barishal', 'Sylhet', 'Rangpur', 'Mymensingh', 'Patuakhali', 'Cox\'s Bazar']

# Generate sample data
data = {
    'uid': range(1, n_samples + 1),
    'patient_name': [fake.name() for _ in range(n_samples)],  # Generate English names
    'sex': np.random.choice(['Male', 'Female'], n_samples),
    'age': np.random.normal(65, 15, n_samples).clip(0, 120),  # Age between 0-120
    'bmi': np.random.normal(25, 8, n_samples).clip(10, 50),   # BMI between 10-50
    'smoke': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),  # 40% smokers
    'location': np.random.choice(locations, n_samples),
    'rs10007052': np.random.normal(0, 1, n_samples),
    'rs9296092': np.random.normal(0, 1, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate target variable (class) with equal contribution from all features
# Normalize all features to 0-1 range
age_factor = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
bmi_factor = (df['bmi'] - df['bmi'].min()) / (df['bmi'].max() - df['bmi'].min())
smoke_factor = (df['smoke'] == 'Yes').astype(int)
location_factor = pd.get_dummies(df['location']).mean(axis=1)  # Convert location to numerical
genetic_factor = (df['rs10007052'] + df['rs9296092']) / 2
genetic_factor = (genetic_factor - genetic_factor.min()) / (genetic_factor.max() - genetic_factor.min())

# Combine all factors with equal weights
prob = (
    0.2 * age_factor +      # Age contribution
    0.2 * bmi_factor +      # BMI contribution
    0.2 * smoke_factor +    # Smoking contribution
    0.2 * location_factor + # Location contribution
    0.2 * genetic_factor    # Genetic markers contribution
)

# Normalize probability to 0-1 range
prob = (prob - prob.min()) / (prob.max() - prob.min())

# Generate class based on probability
df['class'] = (prob > 0.5).astype(int)

# Save to CSV
df.to_csv('data/COPD_Data.csv', index=False)

print("Sample dataset generated successfully!")
print(f"Dataset shape: {df.shape}")
print("\nClass distribution:")
print(df['class'].value_counts(normalize=True))
print("\nFeature correlations with COPD:")
correlations = df.drop(['uid', 'patient_name', 'class'], axis=1).apply(
    lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x
).corrwith(df['class'])
print(correlations.sort_values(ascending=False))
print("\nSample of the data:")
print(df.head()) 