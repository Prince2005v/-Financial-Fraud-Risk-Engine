import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ SETTINGS
# ==========================================
FILE_PATH = 'Fraud.csv' 
SAMPLE_MODE = False      # Set to False to run on the massive 6.3M dataset
SAMPLE_FRACTION = 0.1   # Uses 10% of data for rapid testing

# ==========================================
# 1. LOAD & OPTIMIZE
# ==========================================
print(f"Loading data from {FILE_PATH}...")
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"ERROR: Could not find {FILE_PATH}. Make sure it is saved in the same directory!")
    exit()

if SAMPLE_MODE:
    print(f"SAMPLE MODE ON: Taking a random {SAMPLE_FRACTION*100}% sample for rapid testing.")
    df = pd.concat([
        df[df['isFraud'] == 0].sample(frac=SAMPLE_FRACTION, random_state=42),
        df[df['isFraud'] == 1].sample(frac=SAMPLE_FRACTION, random_state=42)
    ])

start_mem = df.memory_usage().sum() / 1024**2

for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')
    elif df[col].dtype == 'int64':
        df[col] = df[col].astype('int32')
        
print(f"Memory Reduced: {start_mem:.2f} MB -> {df.memory_usage().sum() / 1024**2:.2f} MB\n")

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("Engineering Features...")
df_model = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].reset_index(drop=True)
df_model['errorBalanceOrig'] = df_model['newbalanceOrig'] + df_model['amount'] - df_model['oldbalanceOrg']
df_model['errorBalanceDest'] = df_model['oldbalanceDest'] + df_model['amount'] - df_model['newbalanceDest']

df_model['Orig_Zeroed_Out'] = (df_model['newbalanceOrig'] == 0.0).astype(int)
df_model['Dest_Started_Empty'] = (df_model['oldbalanceDest'] == 0.0).astype(int)
df_model['hour_of_day'] = df_model['step'] % 24

X = df_model.drop(['isFraud', 'type', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore')
y = df_model['isFraud']

# ==========================================
# 3. SPLIT & SCALE
# ==========================================
print("Splitting and Scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. TRAIN XGBOOST
# ==========================================
print("Training XGBoost Engine... (This may take a moment)")
scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_model = XGBClassifier(
    scale_pos_weight=scale_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)

# ==========================================
# 5. EVALUATE
# ==========================================
print("\n" + "="*40)
print("🚀 MODEL TEST RESULTS (XGBoost)")
print("="*40)

y_pred = xgb_model.predict(X_test_scaled)
y_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud (1)']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")

print("\n✔️ Test Complete! The pipeline is stable and ready.")
