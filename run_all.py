from src.preprocessing import load_data, select_features, temporal_split
from src.pipeline import train_random_forest, train_xgboost, train_lightgbm, evaluate_models, plot_roc_pr, shap_analysis
from sklearn.model_selection import TimeSeriesSplit

# Load and preprocess
df = load_data("data/raw/master.csv")
#df = pd.read_csv()
df, features = select_features(df)
X_train, X_test, y_train, y_test, scaler = temporal_split(df, features)

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Train models
rf_search = train_random_forest(X_train, y_train, tscv)
xgb_search = train_xgboost(X_train, y_train, tscv)
lgbm_search = train_lightgbm(X_train, y_train, tscv)

models = {
    "Random Forest": rf_search,
    "XGBoost": xgb_search,
    "LightGBM": lgbm_search
}

# Evaluate
y_preds, comparison_df = evaluate_models(models, X_test, y_test)

# Plot ROC & PR curves
plot_roc_pr(models, X_test, y_test)

# SHAP analysis for XGBoost
shap_analysis(xgb_search.best_estimator_, X_test, features)