# ML_COPD_Project_shadhin


Early Detection of Chronic Obstructive Pulmonary Disease (COPD) Using Machine Learning

Overview
--------
This project presents a machine learning approach for the early detection of Chronic Obstructive Pulmonary Disease (COPD) using a dataset that contains a mix of clinical, demographic, lifestyle, and genetic information. The goal is to predict whether a patient is likely to have COPD, allowing for earlier interventions and improved patient outcomes.

Objective
---------
- To build a classification model that predicts the likelihood of COPD in patients.
- To identify key features (like smoking habits, genetic markers, age, etc.) that contribute to the disease.
- To support medical professionals with data-driven insights for faster and more accurate diagnosis.

Dataset Description
-------------------
Source: https://www.kaggle.com/datasets/mexwell/chronic-obstructive-pulmonary-disorder-copd
Format: CSV

Features Used:
- uid: Unique patient identifier (not used in model)
- label: Diagnostic label or patient group (if present)
- sex: Gender of the patient
- age: Age in years
- bmi: Body Mass Index
- smoke: Smoking status (Yes/No or similar)
- location: Geographic location of the patient
- rs10007052 to rs9296092: Genetic SNP markers
- class: Target variable (1 = COPD, 0 = No COPD)

Project Workflow
----------------
1. Data Preprocessing:
   - Handle missing values
   - Encode categorical features (sex, smoke, location)
   - Drop uid column

2. Feature Engineering:
   - Scale numerical features (age, bmi, SNP values)

3. Model Development:
   - Split data
   - Train models: Logistic Regression, Random Forest, XGBoost, SVM
   - Evaluate using Accuracy, Precision, Recall, F1-Score

4. Optimization and Visualization:
   - Hyperparameter tuning
   - Confusion matrix, ROC curve, feature importance

Technologies Used
-----------------
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn, XGBoost
- Jupyter Notebook

Installation
------------
1. git clone https://github.com/your-username/copd-ml-detection.git
2. cd copd-ml-detection
3. pip install -r requirements.txt
4. jupyter notebook

Results Summary
---------------
- High accuracy in predicting COPD using demographic, clinical, and genetic features.
- Smoking, age, and certain SNPs are key indicators.

Future Improvements
-------------------
- Expand with more genetic data.
- Develop a real-time web prediction tool.
- Experiment with deep learning methods.

License
-------
MIT License

Acknowledgements
----------------
- Kaggle for dataset hosting
- Open-source Python tools and community

