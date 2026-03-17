# src/pipeline.py
import os
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import imblearn

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import ( classification_report, accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score )
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ================================
# Random Forest Training
# ================================
def train_random_forest(X_train, y_train, tscv):
    #rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    pipeline_rf = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    rf_params = {
        'model__n_estimators': [200,300,500],
        'model__max_depth': [5,10,15,None],
        'model__min_samples_split': [2,5,10],
        'model__min_samples_leaf': [1,2,4],
        'model__max_features': ['sqrt','log2']
    }
    rf_search = RandomizedSearchCV(pipeline_rf, rf_params, n_iter=20, scoring='f1', cv=tscv, n_jobs=-1, random_state=42)
    rf_search.fit(X_train, y_train)
    print(f"Best RF Params: {rf_search.best_params_}")
    #return rf_search
    calibrated_rf = CalibratedClassifierCV(
        rf_search.best_estimator_,
        method='sigmoid',
        cv=3
    )

    calibrated_rf.fit(X_train, y_train)

    return calibrated_rf


# ================================
# XGBoost Training
# ================================
def train_xgboost(X_train, y_train, tscv):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    #xgb = XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42)
    pipeline_xgb = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', XGBClassifier(
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    xgb_params = {
        'model__n_estimators':[200,400,600],
        'model__max_depth':[3,5,7],
        'model__learning_rate':[0.01,0.05,0.1],
        'model__subsample':[0.7,0.8,1],
        'model__colsample_bytree':[0.7,0.8,1],
        'model__gamma':[0,1,5]
    }
    xgb_search = RandomizedSearchCV(pipeline_xgb, xgb_params, n_iter=20, scoring='f1', cv=tscv, n_jobs=-1, random_state=42)
    xgb_search.fit(X_train, y_train)
    print(f"Best XGB Params: {xgb_search.best_params_}")
    #return xgb_search
    calibrated_xgb = CalibratedClassifierCV(
        xgb_search.best_estimator_,
        method='sigmoid',
        cv=3
    )

    calibrated_xgb.fit(X_train, y_train)

    return calibrated_xgb

# ================================
# LightGBM Training
# ================================

def train_lightgbm(X_train, y_train, tscv):
    #lgbm = LGBMClassifier(is_unbalance=True, random_state=42)
    pipeline_lgbm = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', LGBMClassifier(is_unbalance=True, random_state=42))
    ])
    lgbm_params = {
        'model__n_estimators':[200,400,600],
        'model__learning_rate':[0.01,0.05,0.1],
        'model__num_leaves':[20,31,50],
        'model__max_depth':[5,10,-1],
        'model__subsample':[0.7,0.8,1],
        'model__colsample_bytree':[0.7,0.8,1]
    }
    lgbm_search = RandomizedSearchCV(pipeline_lgbm, lgbm_params, n_iter=20, scoring='f1', cv=tscv, n_jobs=-1, random_state=42)
    lgbm_search.fit(X_train, y_train)
    print(f"Best LGBM Params: {lgbm_search.best_params_}")
    #return lgbm_search
    calibrated_lgbm = CalibratedClassifierCV(
        lgbm_search.best_estimator_,
        method='sigmoid',
        cv=3
    )

    calibrated_lgbm.fit(X_train, y_train)

    return calibrated_lgbm

# ================================
# Model Evaluation
# ================================

def evaluate_models(models: dict, X_test, y_test, threshold=0.35):
    """Compare multiple models and print metrics."""
    comparison_data = []
    y_preds = {}

    for name, model in models.items():

        y_probs = model.predict_proba(X_test)[:,1]
        y_pred = (y_probs > threshold).astype(int)
        y_preds[name] = y_pred

        comparison_data.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (1)": precision_score(y_test, y_pred),
            "Recall (1)": recall_score(y_test, y_pred),
            "F1-Score (1)": f1_score(y_test, y_pred)
        })

        print(f"\n{name} Classification Report:\n")
        print(classification_report(y_test, y_pred))

    comparison_df = pd.DataFrame(comparison_data).sort_values(by="F1-Score (1)", ascending=False)
    print("\nModel Comparison:\n", comparison_df)
    return y_preds, comparison_df

# ================================
# ROC + PR Curves
# ================================
def plot_roc_pr(models: dict, X_test, y_test, image_dir='images'):
    os.makedirs(image_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{image_dir}/ROC_Curve_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        ap_score = average_precision_score(y_test, y_probs)
        plt.plot(recall, precision, label=f'{name} (AP={ap_score:.2f})')
    baseline = y_test.mean()
    plt.axhline(y=baseline, linestyle='--', color='gray', label=f'Baseline ({baseline:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{image_dir}/Precision_Recall_Curve_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# SHAP Analysis
# ================================

def shap_analysis(model, X_test, model_features, image_dir='images', sample_size=500):
    os.makedirs(image_dir, exist_ok=True)
    X_sample = X_test[:sample_size]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.title('SHAP Value Impact on Model Prediction')
    plt.savefig(f'{image_dir}/Shap_summary.png', dpi=300, bbox_inches='tight')
    shap.summary_plot(shap_values, X_sample, feature_names=model_features)