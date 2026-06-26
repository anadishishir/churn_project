Customer Retention Analytics System 
This project focuses on predicting customer retention for a telecommunications company. By identifying customers likely to leave (churn), businesses can take proactive steps to improve customer loyalty and reduce revenue loss.

Project Overview
I developed an end-to-end Machine Learning pipeline that processes raw customer data, handles class imbalances, and deploys a predictive model via a Flask web application.

Key Highlights:
Problem Type: Binary Classification (Churn vs. No Churn)

Tech Stack: Python, Pandas, Scikit-Learn, XGBoost, FastAPI, Streamlit 

Data Source: Telco Customer Churn dataset 

Technical Workflow 
1. Data Cleaning & Preprocessing 
Type Conversion: Handled numeric conversion for features like TotalCharges, ensuring no data loss from formatting errors. 

Handling Missing Values: Cleaned the dataset by removing null entries to ensure model stability. 

Feature Engineering: Applied OneHotEncoder for categorical variables and StandardScaler to normalize numerical features. 

2. Machine Learning Pipeline 
To ensure the model is robust and handles real-world data issues, I implemented : 

Class Imbalance Handling: Used SMOTE (Synthetic Minority Over-sampling Technique) within an ImbPipeline to address the fact that only ~26% of customers in the dataset had churned. 

Model Selection: Evaluated multiple algorithms, including Logistic Regression, Random Forest, and XGBoost.

3. Model Deployment (FasAPI and Streamlit)
This application is designed to be deployed as two seperate services : the FastAPI backend and Streamlit Frontend. 

Launch Backend : 
python app.py 

Launch Frontend : 
streamlit run frontend.py 

Results & Evaluation
The model was evaluated using industry-standard metrics to ensure high precision and recall.

ROC-AUC Score: Used to measure the model's ability to distinguish between churn and non-churn classes.

Evaluation Strategy: I utilized a Precision-Recall Curve to optimize the threshold for identifying at-risk customers, ensuring a balance between catching true churners and minimizing false alarms.

Repository Structure
churn.ipynb: Full data analysis, visualization, and model training workflow.

app.py: Flask application for real-time predictions.

churn_model.pkl: The saved production-ready model.

requirements.txt: List of dependencies (numpy, pandas, sklearn, xgboost, flask, imblearn).

How to Run
Initialize Enviroment : 
python -m venv venv 
source venv/bin/activate 
pip install -r requirements.txt 

Launch Backend : 
python app.py 

Launch Frontend : 
streamlit run frontend.py 

Personal Learning Notes
Through this project, I gained hands-on experience in:

Building custom Scikit-Learn Pipelines for reproducible data science.

Addressing data leakage and imbalanced datasets using SMOTE.

Bridging the gap between Data Science (Notebooks) and Software Engineering (Flask)  