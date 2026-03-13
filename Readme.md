# Fleet-Asset-Intelligence-System

**Asset Reliability: Fleet Predictive Maintenance Pipeline**

Predictive modeling (XGBoost, Random Forest) to minimize unscheduled downtime and optimize asset lifecycle costs.

End-to-end ML pipeline to predict vehicle component failure within a 30-day window using Time-Series Telemetry, SHAP, and SMOTE.

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

Maintaining a "Healthy" fleet is a balancing act. Inspecting vehicles too often leads to wasted labor; inspecting too late leads to towing fees.

- Low ROI on PM: Sales/Operations losing money when healthy vehicles are pulled for unnecessary checks.
- Safety Risk: Missing critical signals in brake or engine systems.
- Inefficient Labor: Maintenance technicians spending time on low-priority assets while high-risk vehicles remain on the road.

### Tech Stack

- Language: Python 3.11
- Libraries: Pandas, NumPy, Scikit-Learn, Random Forest, XGBoost, Imbalanced-Learn, SHAP
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

The Industry Standard: We must use a Temporal Split (or Time-Series Split).

**3. Feature Engineering:**

Fit Scalers on Train Only.Prevent data leakage.
- Utilization Index: Daily Miles / Average Fleet Miles.
- Sensor Drift: Calculated the slope of Oil Pressure over a 7-day window.

**4. Model Development:**

Train/Tune on Train set. Build the logic.

**Model Evaluation:**

Models were evaluated using:

- ROC-AUC
- Precision-Recall Curve
- Recall (Critical for failure detection). In predictive maintenance, Recall is prioritized to minimize missed failures.
- F1 Score

**5. Model Interpretability (SHAP):**

Used SHAP to prove that Engine_Temperature and Fault_Code_Count are the two highest weighted features.

## Expected Business Benefits

- Reduced Downtime: Expecting 15-20% reduction in unscheduled maintenance events.
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