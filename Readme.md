# Fleet-Asset-Intelligence-System

**Asset Reliability: Fleet Predictive Maintenance Pipeline**

Predictive modeling (XGBoost, Random Forest and LightGBM) to minimize unscheduled downtime and optimize asset lifecycle costs.

End-to-end ML pipeline to predict vehicle component failure within a 30-day window using Time-Series Telemetry, SHAP, and Cost-sensitive learning.

**Domain Expertise**
Having previously worked on the Enterprise Asset Management (EAM) platform at Cetaris, I designed this project to address the specific 'unplanned downtime' challenges faced by fleet managers. I focused on data points like Meter Readings and Failure Codes, which are the standard 'ground truth' in Cetaris databases.

**Executive Summary**
In large-scale fleet operations, unexpected mechanical failures are the primary drivers of inflated operational costs. This project develops a Fleet Asset Intelligence System that transitions maintenance from a reactive "fix-when-fail" model to a proactive "predict-and-prevent" strategy. By leveraging time-series sensor telemetry, the system identifies high-risk assets before they break down, allowing managers to schedule repairs during planned downtime, thereby protecting the asset's total lifecycle value.

**Problem Statement**

Current fleet maintenance protocols often rely on static mileage intervals or reactive responses to dashboard alerts. This results in:

Operational Disruptions: Unscheduled vehicle downtime halting supply chain delivery.

High Emergency Costs: Premium pricing for last-minute parts procurement and emergency labor.

Asset Depreciation: Repeated catastrophic failures reducing the long-term resale value of the fleet.

**Project Objectives**

The goal is to develop a robust classification pipeline capable of predicting a failure event within the next 30 days. The model analyzes a multi-dimensional relational dataset:

- Asset Metadata: Vehicle type (Heavy Duty, Medium Duty, Light Duty), age, and manufacturer specifications.
- Telemetry Trends: Rolling averages of oil pressure, engine temperature, and fuel consumption.
- Usage Metrics: Total odometer mileage, daily utilization rates, and days since last service.
- Historical Reliability: Previous work order counts and total downtime history.

**Business Problem**

Fleet managers often rely on:

 - fixed service intervals
- reactive breakdown repairs

This leads to:

- unexpected downtime
- expensive emergency repairs
- inefficient maintenance scheduling

This project builds a predictive maintenance model that identifies high-risk vehicles before failure occursMaintaining a "Healthy" fleet is a balancing act. Inspecting vehicles too often leads to wasted labor; inspecting too late leads to towing fees.

### Tech Stack

- Language: Python 3.11
- Libraries: Pandas, NumPy, Scikit-Learn, Random Forest, XGBoost, LightGBM, SHAP
- Visualization: Matplotlib, Seaborn 


### System Architecture

Telemetry Data + Work Orders + Asset Metadata
                │
                ▼
        Data Preprocessing
                │
                ▼
        Feature Engineering
    (Rolling statistics, stress indicators)
                │
                ▼
        Temporal Train/Test Split
                │
                ▼
       Machine Learning Models
  (Random Forest, XGBoost, LightGBM)
                │
                ▼
         Model Evaluation
    (ROC-AUC, Precision-Recall)
                │
                ▼
        Model Explainability
             (SHAP)

### The Data Science Workflow

**1. Data Engineering (Relational Logic):**

Generated a high-fidelity synthetic dataset mimicking enterprise FMMS (Fleet Maintenance Management Systems).

Tables

    1. Assets : Vehicle metadata.
       - asset_id , asset_type, purchase_year

    2. Telemetry : Daily sensor readings.
       - odometer, vibration_index, oil_pressure, engine_load, coolant_temp, ambient_temp, daily_utilization

    3. Work Order : Failure events
       - asset_id, date, error_type

Target Definition

The model predicts: Failure within the next 30 days

    - target = 1 if failure occurs within next 30 days
    - target = 0 otherwise

**2. Exploratory Data Analysis (EDA):**

- Discovered that "Days Since Last Service" is a non-linear predictor—risk spikes exponentially after 180 days.
- Identified Data Leakage: Removed "Repair Cost" from the training set, as it is only known after a failure occurs.

**3. Feature Engineering:**

Fit Scalers on Train Only.Prevent data leakage.
- rolling window features.
- Sensor Drift: Calculated the slope of Vibration index and coolent temp over days approching to failure.
- Rolling time-series features capture mechanical degradation: sensor volatility, thermal stress, usage intensity and historical  reliability.

**4. Temporal Split(or Time-Series Split)**

Split by Date : Benchmarked models using a Time-Series Split (Walk-forward validation) to ensure the model isn't "seeing the future."

The "Fleet Data" Twist: Time-Series Splitting
In a fleet project, we cannot use a standard random train_test_split.

Reason: If we have 30 days of data for a truck, and randomly put Day 5 in the test set and Day 6 in the training set, model is essentially "predicting the past using the future."


**5. Machine Learning Models : Hyper parameter tunning and Model Development:**

| Model | Purpose |
| :--: | :-- |
| Random Forest | baseline nonlinear model |
| XGBoost | gradient boosting model |
| LightGBM | optimized boosting model |

Class imbalance handled using cost-sensitive learning instead of SMOTE.

**Model Evaluation:**

Models were evaluated using:

- ROC-AUC
- Precision-Recall Curve
- Recall (Critical for failure detection). In predictive maintenance, Recall is prioritized to minimize missed failures.
- F1 Score
- Precision

**5. Model Interpretability (SHAP):**

SHAP analysis was used to explain predictions.

Example insights:

 - High vibration volatility increases failure probability
 - Thermal stress contributes to component degradation
 - Assets overdue for service have higher risk.

## Expected Business Impact

    - Reduced Downtime: Expecting 15-20% reduction in unscheduled maintenance events.
    - Cost Savings: Lower "Cost per Mile" by avoiding emergency road-call fees.
    - Labor Optimization: Better scheduling of shop technicians based on predicted demand.
    - Improved fleet lifecycle management

**Folder Structure**

fleet-asset-intelligence/

data/
images/
notebooks/
    01_data_generation.ipynb
    02_merging_eda_featureeng.ipynb
    03_split_model_evaluation_SHAP.ipynb
models/
README.md
requirements.txt


fleet-asset-intelligence/

├── data/               # Raw Assets, Telemetry and Work Order CSVs
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