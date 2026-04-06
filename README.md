# 🏦 Financial Fraud Detection Project

This folder contains the complete, production-ready machine learning pipeline for detecting fraudulent financial transactions.

## Files Provided
- `test_fraud_model.py`: An end-to-end Python script that loads data, optimizes memory, engineers features, trains an XGBoost model, and provides precision/recall metrics.
- `Fraud.csv`: (You will manually need to place your Kaggle dataset here).

## How to Test
1. Make sure you place `Fraud.csv` exactly in this folder.
2. The script defaults to `SAMPLE_MODE = True` (using 10% of the dataset) to avoid burning your CPU memory while testing.
3. Open your terminal in this directory and execute:
   ```bash
   python test_fraud_model.py
   ```
