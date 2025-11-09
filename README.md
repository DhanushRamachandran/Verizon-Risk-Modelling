# Verizon Delinquency Risk Modelling


This project focuses on predicting high-risk customers in Verizon’s device financing plans using advanced machine learning and cloud automation. The pipeline integrates AWS, MLflow, and XGBoost ensembles to build a scalable, production-ready delinquency risk model.

Telecom companies face high risk from customers defaulting on device financing plans. This project aims to predict delinquent customers early, enabling proactive actions such as revised terms, reminders, or alternate financing routes.

Key goals:

1. Build an accurate and explainable delinquency prediction model

2. Automate the ML lifecycle (data ingestion → training → tracking → deployment)

3. Enable reproducibility and monitoring through MLflow
# Features:
 XGBoost Ensemble Model — Optimized to capture complex nonlinear relationships and reduce false negatives

☁️AWS ML Pipelines — Automated preprocessing, feature engineering, and training via AWS S3, Lambda, and SageMaker

📈MLflow Integration — Experiment tracking, versioning, and model registry for reproducibility

🔍Feature Importance Analysis — Insights into behavioral and payment variables driving delinquency

📊18% Improvement in prediction accuracy and recall over baseline
