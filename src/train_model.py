from preprocess import (
    X_train_processed,
    X_test_processed,
    y_train,
    y_test,
    preprocessor
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

import joblib


# ============================================================
# 1. TRAIN / VALIDATION SPLIT
# ============================================================

X_train_model, X_val, y_train_model, y_val = train_test_split(
    X_train_processed,
    y_train,
    test_size=0.125,
    stratify=y_train,
    random_state=42
)

print("\nTrain samples:", X_train_model.shape[0])
print("Validation samples:", X_val.shape[0])
print("Test samples:", X_test_processed.shape[0])


# ============================================================
# 2. LOGISTIC REGRESSION
# ============================================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_processed,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test_processed
)

logistic_proba = logistic_model.predict_proba(
    X_test_processed
)[:, 1]

logistic_precision = precision_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_proba
)

print("Precision:", logistic_precision)
print("Recall:", logistic_recall)
print("F1 Score:", logistic_f1)
print("ROC-AUC:", logistic_roc_auc)

print("\nLogistic Regression Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_pred
    )
)


# ============================================================
# 3. RANDOM FOREST
# ============================================================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")

rf_model.fit(
    X_train_processed,
    y_train
)

rf_pred = rf_model.predict(
    X_test_processed
)

rf_proba = rf_model.predict_proba(
    X_test_processed
)[:, 1]

rf_precision = precision_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_roc_auc = roc_auc_score(
    y_test,
    rf_proba
)

print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
print("ROC-AUC:", rf_roc_auc)

print("\nRandom Forest Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)


# ============================================================
# 4. XGBOOST
# ============================================================

print("\n==============================")
print("XGBOOST")
print("==============================")

scale_pos_weight = (
    (y_train_model == 0).sum()
    /
    (y_train_model == 1).sum()
)

print(
    "Scale Pos Weight:",
    scale_pos_weight
)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost...")

xgb_model.fit(
    X_train_model,
    y_train_model,
    eval_set=[
        (X_val, y_val)
    ],
    verbose=False
)


# ============================================================
# 5. THRESHOLD OPTIMIZATION
# ============================================================

print("\n==============================")
print("THRESHOLD OPTIMIZATION")
print("==============================")

xgb_val_proba = xgb_model.predict_proba(
    X_val
)[:, 1]

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

best_threshold = 0.50
best_f1 = 0.0

for threshold in thresholds:

    val_pred = (
        xgb_val_proba >= threshold
    ).astype(int)

    val_precision = precision_score(
        y_val,
        val_pred,
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        val_pred,
        zero_division=0
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    print(
        "Threshold:",
        threshold,
        "| Precision:",
        val_precision,
        "| Recall:",
        val_recall,
        "| F1:",
        val_f1
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_threshold = threshold

print("\nBest Threshold:", best_threshold)
print("Best Validation F1:", best_f1)


# ============================================================
# 6. FINAL EVALUATION ON TEST
# ============================================================

print("\n==============================")
print("FINAL XGBOOST EVALUATION")
print("==============================")

xgb_test_proba = xgb_model.predict_proba(
    X_test_processed
)[:, 1]

xgb_pred_default = (
    xgb_test_proba >= 0.50
).astype(int)

xgb_pred_optimized = (
    xgb_test_proba >= best_threshold
).astype(int)


# ------------------------------------------------------------
# Default threshold = 0.50
# ------------------------------------------------------------

default_precision = precision_score(
    y_test,
    xgb_pred_default,
    zero_division=0
)

default_recall = recall_score(
    y_test,
    xgb_pred_default,
    zero_division=0
)

default_f1 = f1_score(
    y_test,
    xgb_pred_default,
    zero_division=0
)

default_roc_auc = roc_auc_score(
    y_test,
    xgb_test_proba
)

print("\nXGBoost - Default Threshold 0.50")
print("Precision:", default_precision)
print("Recall:", default_recall)
print("F1 Score:", default_f1)
print("ROC-AUC:", default_roc_auc)


# ------------------------------------------------------------
# Optimized threshold
# ------------------------------------------------------------

optimized_precision = precision_score(
    y_test,
    xgb_pred_optimized,
    zero_division=0
)

optimized_recall = recall_score(
    y_test,
    xgb_pred_optimized,
    zero_division=0
)

optimized_f1 = f1_score(
    y_test,
    xgb_pred_optimized,
    zero_division=0
)

print("\nXGBoost - Optimized Threshold")
print("Threshold:", best_threshold)
print("Precision:", optimized_precision)
print("Recall:", optimized_recall)
print("F1 Score:", optimized_f1)

print("\nOptimized Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        xgb_pred_optimized
    )
)


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print("\nLogistic Regression:")
print("Precision:", logistic_precision)
print("Recall:", logistic_recall)
print("F1:", logistic_f1)
print("ROC-AUC:", logistic_roc_auc)

print("\nRandom Forest:")
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1:", rf_f1)
print("ROC-AUC:", rf_roc_auc)

print("\nXGBoost Default:")
print("Precision:", default_precision)
print("Recall:", default_recall)
print("F1:", default_f1)
print("ROC-AUC:", default_roc_auc)

print("\nXGBoost Optimized:")
print("Threshold:", best_threshold)
print("Precision:", optimized_precision)
print("Recall:", optimized_recall)
print("F1:", optimized_f1)
print("ROC-AUC:", default_roc_auc)


# ============================================================
# 8. SAVE MODEL
# ============================================================

joblib.dump(
    xgb_model,
    "models/xgboost_fraud_model.pkl"
)

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print("\nXGBoost model saved successfully!")
print("Preprocessor saved successfully!")