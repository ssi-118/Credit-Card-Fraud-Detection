# Credit Card Fraud Detection System

A Machine Learning based web application that detects fraudulent credit card transactions using transaction behavior, customer details, and geographical analysis. The project provides real-time fraud prediction with interactive visualizations and risk simulation.

---

## Features

- Real-time fraud prediction
- Fraud probability score
- What-If transaction simulation
- Risk radar chart visualization
- Distance-based fraud analysis
- Interactive Streamlit dashboard

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly

---

## Dataset

The project uses a Credit Card Fraud Detection dataset from Kaggle:
```
https://www.kaggle.com/datasets/kartik2112/fraud-detection
```
It includes features like: 
- Transaction amount
- Merchant category
- Customer demographics
- Transaction time
- Geographic coordinates
- Fraud labels

Additional engineered features:

- `amount_log`
- `distance_km`
- `is_weekend`

---

## Algorithms Used

- Logistic Regression
- Random Forest
- XGBoost

## Best Performing Model
**XGBoost**

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

## Install Dependencies
```
pip install -r requirements.txt
```

## Run Application
```
streamlit run src/app.py
```

## Live Demo
```
https://credit-card-fraud-detection-exvtaxtdggm49pynn4cxzn.streamlit.app/
```
