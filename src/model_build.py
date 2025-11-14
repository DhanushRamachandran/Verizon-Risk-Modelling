import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample
from matplotlib import pyplot as plt

df = pd.read_csv("data/Verizon_Data.csv")

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# build and eval logistic regression model
lr = LogisticRegression( penalty="l2",max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Logistic Regression Confusion Matrix:")
cm=(confusion_matrix(y_test, y_pred_lr))
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
lr_probs = lr.predict_proba(X_test)[:, 1]

lr_auc = roc_auc_score(y_test,lr_probs)
print(f"Logistic Regression AUC: {lr_auc:.3f}")
# save model
lr_model_path = "model/logistic_regression_model.pkl"
import joblib
joblib.dump(lr, lr_model_path)
# change the threshold for logistic regression
threshold = 0.95
y_pred_threshold = (lr_probs >= threshold).astype(int)
print(f"Logistic Regression Classification Report at threshold {threshold}:")
print(classification_report(y_test, y_pred_threshold))
# confusion matrix
cm_threshold = confusion_matrix(y_test, y_pred_threshold)
cm_display_threshold = ConfusionMatrixDisplay(cm_threshold).plot()
plt.show()






# ------------------------------------------------------------------

# build and eval random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200,300],
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
cm=(confusion_matrix(y_test, y_pred_rf))
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
rf_probs = best_rf.predict_proba(X_test)[:, 1] 

rf_auc = roc_auc_score(y_test, rf_probs)

print(f"Random Forest AUC: {rf_auc:.3f}")

# save model
rf_model_path = "model/random_forest_model.pkl"
import joblib
joblib.dump(best_rf, rf_model_path)

# change the threshold for random forest
threshold = 0.8
y_pred_threshold = (rf_probs >= threshold).astype(int)
print(f"Logistic Regression Classification Report at threshold {threshold}:")
print(classification_report(y_test, y_pred_threshold))
# confusion matrix
cm_threshold = confusion_matrix(y_test, y_pred_threshold)
cm_display_threshold = ConfusionMatrixDisplay(cm_threshold).plot()
plt.show()

# -----------------------------------------------------
from xgboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Base XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Hyperparameter grid
param_grid_xgb = {
    'n_estimators': [100, 200,300],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05],
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
y_pred_xgb = best_xgb.predict(X_test)
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

print("XGBoost Confusion Matrix:")
cm=(confusion_matrix(y_test, y_pred_xgb))
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
# Probabilities for AUC

xgb_probs = best_xgb.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_probs)
print(classification_report(y_test, y_pred_xgb))
print(f"XGBoost AUC: {xgb_auc:.3f}") 

plt.figure()
plt.plot(*roc_curve(y_test, rf_probs)[:2], label='Random Forest', color='darkorange')
plt.plot(*roc_curve(y_test, lr_probs)[:2], label='Logistic Regression', color='blue')
plt.plot(*roc_curve(y_test, xgb_probs)[:2], label='XGBoost', color='green')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# change the threshold for XGB
threshold = 0.6
y_pred_threshold = (xgb_probs >= threshold).astype(int)
print(f"XGB Regression Classification Report at threshold {threshold}:")
print(classification_report(y_test, y_pred_threshold))
print(confusion_matrix(y_test, y_pred_threshold))

# plot probability distributions
plt.figure(figsize=(10, 6))
plt.hist(rf_probs[y_test == 0], bins=25, alpha=0.5, label='Random Forest - Class 0', color='blue', density=True)
plt.hist(rf_probs[y_test == 1], bins=25, alpha=0.5, label='Random Forest - Class 1', color='orange', density=True)
plt.hist(lr_probs[y_test == 0], bins=25, alpha=0.5, label='Logistic Regression - Class 0', color='green', density=True)
plt.hist(lr_probs[y_test == 1], bins=25, alpha=0.5, label='Logistic Regression - Class 1', color='red', density=True)
plt.hist(xgb_probs[y_test == 0], bins=25, alpha=0.5, label='XGBoost - Class 0', color='purple', density=True)
plt.hist(xgb_probs[y_test == 1], bins=25, alpha=0.5, label='XGBoost - Class 1', color='brown', density=True)
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Predicted Probability Distributions by Model and Class') 
plt.legend()
plt.show()



# print threshold vs 
#save model
xgb_model_path = "model/xgboost_model.pkl"
import joblib
joblib.dump(best_xgb, xgb_model_path)
