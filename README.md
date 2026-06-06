# 📉 Telecom Customer Churn Prediction: A Machine Learning Approach

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.2+-150458.svg)

## 📌 The Business Problem
Customer churn is one of the highest expenses for telecommunication companies. Acquiring a new customer is significantly more expensive than retaining an existing one. 

The goal of this project is to build a Machine Learning classification model that analyzes customer demographics, account information, and service usage to accurately predict which customers are at high risk of leaving. By identifying these customers early, companies can implement targeted retention strategies (like promotional discounts) to save revenue.

## 🧠 Technical Approach & Methodology

### 1. Data Engineering & Preprocessing
* **Data Source:** IBM Telco Customer Churn Dataset (7,000+ records).
* **Cleaning:** Handled hidden null values in numerical columns caused by text-coercion errors for new customers.
* **Feature Engineering:** Applied **One-Hot Encoding** and **Binary Mapping** to convert complex categorical text (like `InternetService_FiberOptic` and `Contract_Month-to-month`) into a fully numerical matrix.

### 2. Tackling Class Imbalance with SMOTE
In the real world, most customers stay, creating an imbalanced dataset (~73% Stay vs. 27% Churn). A standard model would become heavily biased toward predicting "Stay." 
* **Solution:** Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** strictly on the training data to mathematically generate synthetic "Churn" examples, achieving a perfectly balanced 50/50 training distribution without bleeding data into the test set.

### 3. Threshold Optimization for High Recall
The baseline Random Forest Classifier achieved an initial Recall of 53%, meaning it missed nearly half of the leaving customers.
* **The Optimization:** Since the business cost of a False Negative (losing a high-paying customer) is far greater than a False Positive (accidentally giving a discount to a happy customer), I extracted the raw probability scores and **lowered the decision threshold from 0.50 to 0.30**.
* **The Result:** This optimization spiked the model's **Recall to 81%**, successfully catching the vast majority of at-risk customers.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** NumPy, Pandas
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Environment:** Jupyter Notebook / Visual Studio Code

## 🚀 How to Run Locally
1. Clone this repository: `git clone https://github.com/tuneer-kar/telecom-churn.git`
2. Install the required dependencies: `pip install pandas numpy scikit-learn imbalanced-learn`
3. Run the Jupyter Notebook `01_eda_and_cleaning.ipynb` to see the complete data pipeline and model training.
4. The finalized model is exported as `churn_model.pkl` for future web deployment.

## 📈 Future Scope
* Build an interactive front-end dashboard using **Streamlit** to allow sales managers to input customer data and receive real-time churn probability scores.
* Experiment with **XGBoost** to squeeze out higher precision without sacrificing recall.