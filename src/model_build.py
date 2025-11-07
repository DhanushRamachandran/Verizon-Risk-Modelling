import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample
import math

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# build and eval logistic regression model
lr = LogisticRegression( max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_train)
print("Logistic Regression Classification Report:")
print(classification_report(y_train, y_pred_lr))
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_lr))

lr_probs = lr.predict_proba(X_train)[:, 1]

lr_auc = roc_auc_score(y_train, lr_probs)
print(f"Logistic Regression AUC: {lr_auc:.3f}")
# ------------------------------------------------------------------


# build and eval random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5,10],
    'min_samples_leaf': [1, 2,5],
    'ccp_alpha': [ 0.01,0.02,0.03]
}

grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose =2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Hyperparameters for Random Forest:", best_params)
y_pred_rf = best_rf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf)) 
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
rf_probs = best_rf.predict_proba(X_test)[:, 1] 

rf_auc = roc_auc_score(y_test, rf_probs)

print(f"Random Forest AUC: {rf_auc:.3f}")

# -----------------------------------------------------
from xgboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Base XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Hyperparameter grid
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

# Grid Search
grid_search_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_xgb,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Fit model
grid_search_xgb.fit(X_train,y_train)

# Best model and params
best_xgb = grid_search_xgb.best_estimator_
best_params_xgb = grid_search_xgb.best_params_
print("Best Hyperparameters for XGBoost:", best_params_xgb)

# Predictions
y_pred_xgb = best_xgb.predict(X_train)
print("XGBoost Classification Report:")
print(classification_report(y_train, y_pred_xgb))

print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_xgb))

# Probabilities for AUC
xgb_probs = best_xgb.predict_proba(X_train)[:, 1]
xgb_auc = roc_auc_score(y_train, xgb_probs)
print(f"XGBoost AUC: {xgb_auc:.3f}") 
