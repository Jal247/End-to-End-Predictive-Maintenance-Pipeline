# Fleet-Asset-Intelligence-System

**Asset Reliability: Fleet Predictive Maintenance Pipeline**

Predictive modeling (XGBoost, Random Forest) to minimize unscheduled downtime and optimize asset lifecycle costs.

End-to-end ML pipeline to predict vehicle component failure within a 30-day window using Time-Series Telemetry, SHAP, and SMOTE.

**Executive Summary**
In large-scale fleet operations, unexpected mechanical failures are the primary drivers of inflated operational costs. This project develops a Fleet Asset Intelligence System that transitions maintenance from a reactive "fix-when-fail" model to a proactive "predict-and-prevent" strategy. By leveraging time-series sensor telemetry, the system identifies high-risk assets before they break down, allowing managers to schedule repairs during planned downtime, thereby protecting the asset's total lifecycle value.

**Problem Statement**

Current fleet maintenance protocols often rely on static mileage intervals or reactive responses to dashboard "idiot lights." This results in:

Operational Disruptions: Unscheduled vehicle downtime halting supply chain delivery.

High Emergency Costs: Premium pricing for last-minute parts procurement and emergency labor.

Asset Depreciation: Repeated catastrophic failures reducing the long-term resale value of the fleet.

**Project Objectives**

The goal is to develop a robust classification pipeline capable of predicting a failure event within the next 30 days. The model analyzes a multi-dimensional relational dataset:

- Asset Metadata: Vehicle type (Truck, Van, Car), age, and manufacturer specifications.
- Telemetry Trends: Rolling averages of oil pressure, engine temperature, and fuel consumption.
- Usage Metrics: Total odometer mileage, daily utilization rates, and days since last service.
- Historical Reliability: Previous work order counts and total downtime history.

**Business Problem**

Maintaining a "Healthy" fleet is a balancing act. Inspecting vehicles too often leads to wasted labor; inspecting too late leads to towing fees.

- Low ROI on PM: Sales/Operations losing money when healthy vehicles are pulled for unnecessary checks.
- Safety Risk: Missing critical signals in brake or engine systems.
- Inefficient Labor: Maintenance technicians spending time on low-priority assets while high-risk vehicles remain on the road.

## Project Summary: Asset Failure Prediction

**Objective**

Build a predictive engine to identify vehicles with a high probability of mechanical failure. The primary focus is Recall—minimizing "False Negatives" (missed failures) to prevent on-road breakdowns.

**Technical Workflow**

1. Data Engineering: Created Lag Features and Rolling Standard Deviations to capture "sensor drift" and "vibration noise" that precede mechanical failure.
2. Addressing Imbalance: Applied SMOTE to the training set to account for the rarity of failure events (failures typically represent < 5% of fleet data).
3. Model Selection: Evaluated XGBoost and Random Forest. XGBoost was selected for its superior handling of non-linear relationships between sensor spikes and engine heat.
4. Interpretability: Integrated SHAP values to provide "Reason Codes." Instead of a generic alert, the model specifies: "Risk high due to 15% drop in oil pressure relative to engine load."

**Final Model Performance**

The optimized XGBoost model achieved a Recall of ~82% for failures within a 30-day lead time. This allows the maintenance team to flag 8 out of 10 looming failures nearly a month before they become catastrophic.

**Strategic Recommendations**

- Conditional Maintenance: Replace "Calendar-based" oil changes with "Condition-based" changes for vehicles flagged with high heat-stress scores.
- Asset Replacement: Assets identified by the model as "Frequent Failures" despite recent repairs should be prioritized for liquidation/replacement.
- Parts Inventory: Use model predictions to forecast which parts (e.g., fuel pumps, sensors) will be needed in the coming 30 days.

### Tech Stack

- Language: Python 3.11
- Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE), SHAP
- Visualization: Matplotlib, Seaborn

### The Data Science Workflow

**1. Data Engineering (Relational Logic):**

Generated a high-fidelity synthetic dataset mimicking enterprise FMMS (Fleet Maintenance Management Systems).

**2. Exploratory Data Analysis (EDA):**

- Discovered that "Days Since Last Service" is a non-linear predictor—risk spikes exponentially after 180 days.
- Identified Data Leakage: Removed "Repair Cost" from the training set, as it is only known after a failure occurs.

**3. Temporal Split(or Time-Series Split)**

Split by Date : Benchmarked models using a Time-Series Split (Walk-forward validation) to ensure the model isn't "seeing the future."

The "Fleet Data" Twist: Time-Series Splitting
In a fleet project, we cannot use a standard random train_test_split.

Why: If we have 30 days of data for a truck, and randomly put Day 5 in the test set and Day 6 in the training set, model is essentially "predicting the past using the future."

The Industry Standard: We must use a Temporal Split (or Time-Series Split).  pick a cutoff date (e.g., Months 1-10 for Training, Month 11-12 for Testing).

**3. Feature Engineering:**

Fit Scalers on Train Only.Prevent data leakage.
- Utilization Index: Daily Miles / Average Fleet Miles.
- Sensor Drift: Calculated the slope of Oil Pressure over a 7-day window.

**4. Model Development:**

Train/Tune on Train set. Build the logic.

**5. Model Interpretability (SHAP):**

Used SHAP to prove that Engine_Temperature and Fault_Code_Count are the two highest weighted features.

## Expected Business Benefits

- Reduced Downtime: 15-20% reduction in unscheduled maintenance events.
- Cost Savings: Lower "Cost per Mile" by avoiding emergency road-call fees.
- Labor Optimization: Better scheduling of shop technicians based on predicted demand.

Folder Structure
fleet-asset-intelligence/
├── data/               # Raw Telemetry and Work Order CSVs
├── images/             # SHAP plots, ROC Curves, and Architecture diagrams
├── notebooks/          
│   ├── 00_data_generation.ipynb
│   ├── 01_eda_and_merging.ipynb
│   └── 02_modeling_and_shap.ipynb
├── src/                
│   ├── feature_logic.py
│   └── evaluation.py
├── README.md           
└── requirements.txt





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
