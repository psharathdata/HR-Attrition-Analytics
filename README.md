# Employee Attrition Analytics & Prediction

End-to-end HR analytics project focused on analyzing employee attrition patterns and predicting employees who are at high risk of leaving the company using Machine Learning, SQL, and Interactive Dashboards.

---

# Business Problem

Employee attrition leads to increased hiring costs, reduced productivity, and loss of experienced talent.  
This project helps HR teams identify the key reasons behind employee attrition and predict high-risk employees for proactive retention strategies.

---

# Objectives

- Analyze employee attrition trends
- Identify factors influencing attrition
- Build predictive Machine Learning models
- Generate business insights for HR teams
- Develop interactive dashboards for decision-making

---

# Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SQL
- Tableau
- Power BI
- Streamlit
- SHAP
- Matplotlib
- Plotly

---

# Dataset

Dataset used: IBM HR Analytics Employee Attrition Dataset

The dataset contains:
- Employee demographics
- Salary details
- Job roles
- Department information
- Overtime records
- Work experience
- Job satisfaction
- Attrition status

---

# Project Workflow

## 1. Data Cleaning
- Missing value handling
- Feature engineering
- Data preprocessing
- Encoding categorical variables

## 2. Exploratory Data Analysis (EDA)
- Attrition by department
- Salary vs attrition
- Overtime impact
- Age group analysis
- Job satisfaction trends

## 3. SQL Analysis
Business queries performed:
- Attrition by department
- Attrition by salary band
- Overtime impact analysis
- Employee retention trends
- Job role analysis

## 4. Machine Learning
Models Used:
- Logistic Regression
- Random Forest
- XGBoost

## 5. Explainability
- SHAP feature importance analysis
- High-risk employee identification

## 6. Dashboard Development
- Tableau dashboard
- Power BI dashboard
- Streamlit application

---

# Key Insights

- Employees working overtime showed significantly higher attrition risk
- Lower salary bands had increased employee turnover
- Younger employees were more likely to leave
- Job satisfaction strongly influenced employee retention
- Employees with fewer years at the company showed higher attrition probability

---

# Machine Learning Performance

| Model | ROC-AUC Score |
|---|---|
| Logistic Regression | 0.83 |
| Random Forest | 0.96 |
| XGBoost | 0.97 |

---

# Features

- Employee attrition prediction
- Risk score generation
- High-risk employee identification
- Interactive HR dashboards
- SHAP explainability
- Business intelligence reporting

---

# Dashboard Preview

## Dashboard Includes:
- Total Employees KPI
- Attrition Rate
- High-Risk Employees
- Department Analysis
- Salary Band Analysis
- Overtime Impact
- Risk Distribution
- Interactive Filters

---

# Project Structure

```text
hr-attrition-analysis/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── notebooks/
│   └── hr_attrition_analysis.ipynb
│
├── sql/
│   └── hr_attrition_queries.sql
│
├── dashboard/
│   └── tableau_dashboard.png
│
├── outputs/
│   ├── charts/
│   ├── shap_summary.png
│   └── high_risk_employees.csv
│
├── app.py
├── requirements.txt
└── README.md
