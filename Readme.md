Fleet-Asset-Intelligence-System
An Enterprise-Grade Predictive Maintenance Engine for Asset Lifecycle Management

📊 Executive Summary
This project simulates a high-fidelity Fleet Management System. Unlike standard ML projects that use flat datasets, this system utilizes a relational data structure (Assets, Telemetry, and Work Orders) to predict component failures before they occur. By moving from reactive to proactive maintenance, this pipeline aims to reduce vehicle downtime by an estimated 15-20% and optimize labor allocation.

🛠️ Tech Stack
Language: Python 3.x

Libraries: Pandas, NumPy (Data Engineering), Scikit-Learn, XGBoost (Modeling), SHAP (Model Explainability), Matplotlib/Seaborn (Visualization).

Methodology: CRISP-DM

🏗️ Data Architecture
The project utilizes three logically linked data entities:

Assets: Static metadata for 1,000+ vehicles (Type, Purchase Year, Manufacturer).

Telemetry: Time-series sensor data (Odometer, Oil Pressure, Engine Temp, Fuel Consumption).

Work Orders: Historical maintenance logs used as the "Ground Truth" for training.

🚀 Key Features
Temporal Feature Engineering: Implements rolling averages and "Lag" features to capture sensor degradation trends over time.

Lead-Time Prediction: Instead of predicting if a failure happened, the model predicts the probability of failure within the next 30-day window.

Model Explainability: Uses SHAP values to provide "Reason Codes" for every maintenance alert, allowing mechanics to understand why a vehicle is flagged.

📈 Project Roadmap
[x] Phase 0: Synthetic Data Generation (Relational Logic)

[ ] Phase 1: Exploratory Data Analysis (EDA) & Trend Identification

[ ] Phase 2: Feature Engineering & Class Imbalance Handling

[ ] Phase 3: Model Development (XGBoost / Random Forest)

[ ] Phase 4: Interpretation & Maintenance Strategy
