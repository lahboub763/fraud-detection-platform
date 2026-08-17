# 🛡️ Fraud Detection Platform

An end-to-end Machine Learning platform for detecting potentially fraudulent transactions.

## 📌 Project Overview

This project uses Machine Learning to classify financial transactions as:

- Fraud
- Not Fraud

The platform includes data preprocessing, model training, model evaluation, a FastAPI prediction API, a Streamlit dashboard, and Docker deployment.

## 🏗️ Architecture

```text
Transaction
     ↓
Preprocessing
     ↓
XGBoost Model
     ↓
Fraud Probability
     ↓
FastAPI
     ↓
Streamlit Dashboard