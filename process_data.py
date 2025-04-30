import pandas as pd
from sklearn.impute import SimpleImputer
import os
import matplotlib.pyplot as plt

# Load the data using proper path handling
file_path = os.path.join('dataset', 'finalalldata (1).csv')
df = pd.read_csv(file_path)

# Step 1: Drop Unnecessary Columns
# Original columns in the dataset
original_columns = df.columns.tolist()

# Drop 'uid' and 'class' as they are not needed for modeling
df_step1 = df.drop(columns=['uid', 'class'])

# Show updated columns
updated_columns = df_step1.columns.tolist()

print("Original columns:", original_columns)
print("\nUpdated columns:", updated_columns)

# Step 2: Handle Missing Values
# Check how many missing values are in each column
missing_summary = df_step1.isnull().sum()

# Calculate percentage of missing values per column
missing_percentage = (missing_summary / len(df_step1)) * 100

# Combine into one table
missing_report = pd.DataFrame({
    'Missing Values': missing_summary,
    'Percentage': missing_percentage
})

# Filter only columns with missing data
missing_report = missing_report[missing_report['Missing Values'] > 0]
missing_report = missing_report.sort_values(by='Percentage', ascending=False)

# Show the per-column missing value report
print("\nMissing Value Report (Column-wise):")
print(missing_report)

# Calculate total percentage of missing values in the entire dataset
total_missing = df_step1.isnull().sum().sum()
total_values = df_step1.size
total_missing_percentage = (total_missing / total_values) * 100

print(f"\nTotal Missing Value Percentage in Entire Dataset: {total_missing_percentage:.2f}%")

# Visualize missing values
plt.figure(figsize=(10, 6))
missing_report['Percentage'].plot(kind='bar')
plt.title('Missing Values by Column')
plt.xlabel('Columns')
plt.ylabel('Percentage of Missing Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('missing_values.png')
print("\nMissing values visualization saved as 'missing_values.png'")

# Step 3: Impute Missing Values
# Separate numeric and non-numeric columns
numeric_cols = df_step1.select_dtypes(include=['float64', 'int64']).columns
non_numeric_cols = df_step1.select_dtypes(exclude=['float64', 'int64']).columns

# Impute only numeric columns
imputer = SimpleImputer(strategy='mean')
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_step1[numeric_cols]), columns=numeric_cols)

# Combine back non-numeric columns
df_final = pd.concat([df_numeric_imputed, df_step1[non_numeric_cols].reset_index(drop=True)], axis=1)

# Confirm no missing values
print("\nTotal missing values after imputation:")
print(df_final.isnull().sum().sum())  # Should be 0

# Save the processed data
df_final.to_csv('processed_data.csv', index=False)
print("\nProcessed data saved to 'processed_data.csv'") 