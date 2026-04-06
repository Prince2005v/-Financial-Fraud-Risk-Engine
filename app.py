import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler

# Configure the page
st.set_page_config(page_title="Fraud Detection AI", page_icon="🏦", layout="wide")

st.title("🏦 Financial Fraud Risk Engine")
st.markdown("Enter transaction details below to run an instant real-time AI Risk Assessment.")

# ==========================================
# 1. AI ENGINE BOOTUP 
# ==========================================
@st.cache_resource
def load_ai_engine():
    with st.spinner("Booting AI Engine & Analyzing Historical Data..."):
        # Load a small 1% chunk for rapid web-app training
        df = pd.read_csv('Fraud.csv').sample(frac=0.01, random_state=42)
        
        # Optimize
        for col in df.columns:
            if df[col].dtype == 'float64': df[col] = df[col].astype('float32')
                
        # Engineer Features
        df_model = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].reset_index(drop=True)
        df_model['errorBalanceOrig'] = df_model['newbalanceOrig'] + df_model['amount'] - df_model['oldbalanceOrg']
        df_model['errorBalanceDest'] = df_model['oldbalanceDest'] + df_model['amount'] - df_model['newbalanceDest']
        df_model['Orig_Zeroed_Out'] = (df_model['newbalanceOrig'] == 0.0).astype(int)
        df_model['Dest_Started_Empty'] = (df_model['oldbalanceDest'] == 0.0).astype(int)
        df_model['hour_of_day'] = df_model['step'] % 24
        
        X = df_model.drop(['isFraud', 'type', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore')
        y = df_model['isFraud']

        # Scale & Train
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Guard against zero fraud in 1% sample
        scale_weight = max((len(y) - sum(y)) / max(sum(y), 1), 1)
        
        model = XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler, list(X.columns)

# Boot the engine parameters
model, scaler, feature_names = load_ai_engine()

# ==========================================
# 2. USER INTERFACE
# ==========================================
st.sidebar.header("Transaction Payload")
input_amount = st.sidebar.number_input("Transaction Amount ($)", value=1500.0, step=100.0)
old_bal_orig = st.sidebar.number_input("Sender Old Balance ($)", value=2000.0, step=100.0)
new_bal_orig = st.sidebar.number_input("Sender New Balance ($)", value=500.0, step=100.0)
old_bal_dest = st.sidebar.number_input("Recipient Old Balance ($)", value=0.0, step=100.0)
new_bal_dest = st.sidebar.number_input("Recipient New Balance ($)", value=1500.0, step=100.0)
step_hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 14)

st.sidebar.markdown("---")
if st.sidebar.button("🛡️ Run Security Scan", type="primary"):
    
    # Generate live math features
    err_orig = new_bal_orig + input_amount - old_bal_orig
    err_dest = old_bal_dest + input_amount - new_bal_dest
    zeroed_out = 1 if new_bal_orig == 0.0 else 0
    started_empty = 1 if old_bal_dest == 0.0 else 0
    
    # Package into prediction frame
    payload = pd.DataFrame([[
        input_amount, old_bal_orig, new_bal_orig, old_bal_dest, new_bal_dest,
        err_orig, err_dest, zeroed_out, started_empty, step_hour
    ]], columns=feature_names)
    
    # Predict
    payload_scaled = scaler.transform(payload)
    prob_score = model.predict_proba(payload_scaled)[0][1] * 100
    
    # Display Results
    st.subheader("Action Recommendation")
    
    if prob_score > 85:
        st.error(f"🚨 CRITICAL RISK DETECTED: {prob_score:.2f}% Fraud Probability")
        st.write("**System Action:** Transaction Blocked. Issuing Step-Up Multi-Factor Authentication Push to User.")
    elif prob_score > 50:
        st.warning(f"⚠️ ELEVATED RISK: {prob_score:.2f}% Fraud Probability")
        st.write("**System Action:** Shadow-Queueing. Transaction allowed, but flagged for manual human analyst review.")
    else:
        st.success(f"✅ NORMAL TRANSACTION: {prob_score:.2f}% Risk")
        st.write("**System Action:** Auto-Approved. Millisecond routing applied.")
        
    # Detail Section
    st.markdown("---")
    st.write("**AI Diagnostic Trace:**")
    if err_orig != 0:
        st.write("- ⚠️ **Ledger Anomaly:** Sender metrics mathematically corrupted.")
    if started_empty == 1:
        st.write("- ⚠️ **Burner Profile:** Recipient account was entirely empty upon receiving funds.")
    if zeroed_out == 1:
        st.write("- ⚠️ **Maximum Extraction:** Sender account was forcefully drained to zero.")
