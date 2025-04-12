import joblib
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from sklearn.metrics import r2_score


# Load the saved models
rf_model_with_ii = joblib.load("rf_model_with_ii.pkl")
rf_model_without_ii = joblib.load("rf_model_without_ii.pkl")

fedFundsTest = pd.read_csv("FedfundsTest.csv")
inflationTest = pd.read_csv("InflationTest.csv")
realGDPTest = pd.read_csv("RealGDPTest.csv")
potGDPTest = pd.read_csv("PotGDPTest.csv")
partyData = pd.read_csv("PartyTotal.csv")

fedFundsTest['observation_date'] = pd.to_datetime(fedFundsTest['observation_date'])
inflationTest['observation_date'] = pd.to_datetime(inflationTest['observation_date'])
realGDPTest['observation_date'] = pd.to_datetime(realGDPTest['observation_date'])
potGDPTest['observation_date'] = pd.to_datetime(potGDPTest['observation_date'])
partyData['observation_date'] = pd.to_datetime(partyData['observation_date'], errors='coerce', dayfirst=True)

start_date = '1998-01-01'
end_date = '2006-12-31'

fedFundsTest = fedFundsTest[(fedFundsTest['observation_date'] >= start_date) & (fedFundsTest['observation_date'] <= end_date)]
inflationTest = inflationTest[(inflationTest['observation_date'] >= start_date) & (inflationTest['observation_date'] <= end_date)]
realGDPTest = realGDPTest[(realGDPTest['observation_date'] >= start_date) & (realGDPTest['observation_date'] <= end_date)]
potGDPTest = potGDPTest[(potGDPTest['observation_date'] >= start_date) & (potGDPTest['observation_date'] <= end_date)]
partyData = partyData[(partyData['observation_date'] >= start_date) & (partyData['observation_date'] <= end_date)]


inflationTest['Inflation_Rate'] = (
    (inflationTest['GDPDEF'] / inflationTest['GDPDEF'].shift(4) - 1) * 100
)

merged_pot_gdp = potGDPTest[['observation_date', 'GDPPOT']].merge(
    realGDPTest[['observation_date', 'GDPC1']], on='observation_date'
)

merged_pot_gdp['OutputGap'] = 100 * (
    (np.log(merged_pot_gdp['GDPC1']) - np.log(merged_pot_gdp['GDPPOT']))
)

for lag in range(1, 4):
    fedFundsTest[f'FEDFUNDS_Lag{lag}'] = fedFundsTest['FEDFUNDS'].shift(lag)
    inflationTest[f'Inflation_Rate_Lag{lag}'] = inflationTest['Inflation_Rate'].shift(lag)
    merged_pot_gdp[f'OutputGap_Lag{lag}'] = merged_pot_gdp['OutputGap'].shift(lag)

fedFundsTest = fedFundsTest[fedFundsTest['observation_date'] >= "2000-01-01"]
inflationTest = inflationTest[inflationTest['observation_date'] >= "2000-01-01"]
merged_pot_gdp = merged_pot_gdp[merged_pot_gdp['observation_date'] >= "2000-01-01"]
partyData = partyData[partyData['observation_date'] >= "2000-01-01"]

print(partyData.head())

# Concatenating the DataFrames along columns
merged_test_data = pd.concat(
    [
        fedFundsTest.set_index("observation_date"),
        inflationTest.set_index("observation_date"),
        merged_pot_gdp.set_index("observation_date"),
        partyData.set_index("observation_date")
    ],
    axis=1
).reset_index()

merged_test_data['InflationInteraction'] = merged_test_data['Inflation_Rate'] * merged_test_data['PresidentParty']
merged_test_data['OutputGapInteraction'] = merged_test_data['OutputGap'] * merged_test_data['PresidentParty']

merged_test_data.rename(columns={'FEDFUNDS_Lag1': 'FedFundsLag1', 'Inflation_Rate': 'InflationRate'}, inplace=True)

# Display merged data
print("Merged Test Data:")
print(merged_test_data.head())

print(merged_test_data.columns)

print("Train Model Features:", list(rf_model_with_ii.feature_names_in_))
print("Test Data Columns:", list(merged_test_data.columns))

# Select features to match the trained models
# features_with_ii = ['InflationRate', 'OutputGap', 'PresidentParty',
#                      'InflationInteraction', 'OutputGapInteraction', 'FedFundsLag1']
# features_without_ii = ['InflationRate', 'OutputGap', 'PresidentParty', 'OutputGapInteraction', 'FedFundsLag1']
#
# x_new_with_ii = merged_test_data[features_with_ii]
# x_new_without_ii = merged_test_data[features_without_ii]

features_with_ii = rf_model_with_ii.feature_names_in_
features_without_ii = rf_model_without_ii.feature_names_in_

# ✅ Select features from test dataset
x_new_with_ii = merged_test_data[features_with_ii]
x_new_without_ii = merged_test_data[features_without_ii]

# ✅ Debugging: Check final feature match before prediction
print("Final Test Features (With II):", list(x_new_with_ii.columns))
print("Final Test Features (Without II):", list(x_new_without_ii.columns))

# Make predictions using the models
predictions_with_ii = rf_model_with_ii.predict(x_new_with_ii)
predictions_without_ii = rf_model_without_ii.predict(x_new_without_ii)

# Display predictions
merged_test_data['Predicted_FedFunds_With_II'] = predictions_with_ii
merged_test_data['Predicted_FedFunds_Without_II'] = predictions_without_ii

print(merged_test_data[['observation_date', 'Predicted_FedFunds_With_II', 'Predicted_FedFunds_Without_II']])

# Save predictions to CSV
merged_test_data[['observation_date', 'Predicted_FedFunds_With_II', 'Predicted_FedFunds_Without_II']].to_csv("Predicted_FedFunds.csv", index=False)

# ✅ Load the predicted data
predicted_data = pd.read_csv("Predicted_FedFunds.csv")

# ✅ Load actual Fed Funds rate
actual_data = pd.read_csv("FedfundsTest.csv")  # Ensure this file has the correct actual values

# ✅ Merge actual Fed Funds rate with predictions
merged_data = predicted_data.merge(actual_data[['observation_date', 'FEDFUNDS']], on='observation_date', how='inner')

# ✅ Calculate R² for both models
r2_with_ii = r2_score(merged_data['FEDFUNDS'], merged_data['Predicted_FedFunds_With_II'])
r2_without_ii = r2_score(merged_data['FEDFUNDS'], merged_data['Predicted_FedFunds_Without_II'])

# ✅ Print results
print(f"R² Score (With Inflation Interaction): {r2_with_ii:.4f}")
print(f"R² Score (Without Inflation Interaction): {r2_without_ii:.4f}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Load additional models === #
rf_model_with_lag = joblib.load("rf_model_with_lag.pkl")
rf_model_without_lag = joblib.load("rf_model_without_lag.pkl")

# === Define feature sets === #
features_with_lag = ['InflationRate', 'OutputGap', 'FedFundsLag1']
features_without_lag = ['InflationRate', 'OutputGap']

# === Extract test sets === #
x_test_with_lag = merged_test_data[features_with_lag]
x_test_without_lag = merged_test_data[features_without_lag]
y_test_actual = merged_test_data['FEDFUNDS']

# === Predict with each model === #
y_pred_with_lag = rf_model_with_lag.predict(x_test_with_lag)
y_pred_without_lag = rf_model_without_lag.predict(x_test_without_lag)

# === Define evaluation function === #
def evaluate_rf_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse_mean = mean_squared_error(y_true, y_pred) / np.mean(y_true)
    return {
        'Model': model_name,
        'R²': round(r2, 2),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MSE/Mean': round(mse_mean, 2)
    }

# === Evaluate both models === #
results = []
results.append(evaluate_rf_model(y_test_actual, y_pred_with_lag, "Random Forest **With** Lagged FedFunds"))
results.append(evaluate_rf_model(y_test_actual, y_pred_without_lag, "Random Forest **Without** Lagged FedFunds"))

# === Save results as DataFrame === #
import pandas as pd
rf_test_results = pd.DataFrame(results)
print(rf_test_results)

from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

# ----------------------------- VIF ----------------------------- #
def compute_vif(df, features):
    vif_df = pd.DataFrame()
    vif_df["Feature"] = features
    vif_df["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_df.sort_values(by="VIF", ascending=False)

vif_with_lag = compute_vif(merged_test_data, features_with_lag)
vif_without_lag = compute_vif(merged_test_data, features_without_lag)

print("\nVIF for With Lagged FedFunds:")
print(vif_with_lag)

print("\nVIF for Without Lagged FedFunds:")
print(vif_without_lag)

# ----------------------------- Permutation Importance ----------------------------- #
pfi_with_lag = permutation_importance(rf_model_with_lag, x_test_with_lag, y_test_actual, n_repeats=30, random_state=42)
pfi_without_lag = permutation_importance(rf_model_without_lag, x_test_without_lag, y_test_actual, n_repeats=30, random_state=42)

pfi_df_with_lag = pd.DataFrame({
    "Feature": x_test_with_lag.columns,
    "Importance": pfi_with_lag.importances_mean,
    "StdDev": pfi_with_lag.importances_std
}).sort_values(by="Importance", ascending=False)

pfi_df_without_lag = pd.DataFrame({
    "Feature": x_test_without_lag.columns,
    "Importance": pfi_without_lag.importances_mean,
    "StdDev": pfi_without_lag.importances_std
}).sort_values(by="Importance", ascending=False)

print("\nPermutation Importance for With Lagged FedFunds:")
print(pfi_df_with_lag)

print("\nPermutation Importance for Without Lagged FedFunds:")
print(pfi_df_without_lag)

# ----------------------------- Leave-One-Out (LOO) ----------------------------- #
def loo_analysis(x_test, y_test, model_base, baseline_r2, baseline_rmse, baseline_mae):
    results = []
    for feature in x_test.columns:
        x_test_drop = x_test.drop(columns=[feature])
        model = RandomForestRegressor(
            n_estimators=397,
            max_depth=13,
            min_samples_leaf=4,
            min_samples_split=3,
            random_state=42
        )
        model.fit(x_test_drop, y_test)
        y_pred_drop = model.predict(x_test_drop)

        r2_drop = baseline_r2 - r2_score(y_test, y_pred_drop)
        rmse_drop = np.sqrt(mean_squared_error(y_test, y_pred_drop)) - baseline_rmse
        mae_drop = mean_absolute_error(y_test, y_pred_drop) - baseline_mae

        results.append({
            'Feature': feature,
            'R² Drop': round(r2_drop, 2),
            'RMSE Drop': round(rmse_drop, 2),
            'MAE Drop': round(mae_drop, 2)
        })

    return pd.DataFrame(results).sort_values(by='R² Drop', ascending=False)

baseline_with_r2 = r2_score(y_test_actual, y_pred_with_lag)
baseline_with_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_with_lag))
baseline_with_mae = mean_absolute_error(y_test_actual, y_pred_with_lag)

baseline_without_r2 = r2_score(y_test_actual, y_pred_without_lag)
baseline_without_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_without_lag))
baseline_without_mae = mean_absolute_error(y_test_actual, y_pred_without_lag)

loo_with_lag = loo_analysis(x_test_with_lag, y_test_actual, rf_model_with_lag, baseline_with_r2, baseline_with_rmse, baseline_with_mae)
loo_without_lag = loo_analysis(x_test_without_lag, y_test_actual, rf_model_without_lag, baseline_without_r2, baseline_without_rmse, baseline_without_mae)

print("\nLOO Analysis for With Lagged FedFunds:")
print(loo_with_lag)

print("\nLOO Analysis for Without Lagged FedFunds:")
print(loo_without_lag)

# ----------------------------- OOB Refit ----------------------------- #
# You need to retrain the model on full training data with oob_score=True
rf_oob_with = RandomForestRegressor(
    n_estimators=397, max_depth=13, min_samples_leaf=4, min_samples_split=3,
    oob_score=True, random_state=42
)
rf_oob_with.fit(x_test_with_lag, y_test_actual)

rf_oob_without = RandomForestRegressor(
    n_estimators=397, max_depth=13, min_samples_leaf=4, min_samples_split=3,
    oob_score=True, random_state=42
)
rf_oob_without.fit(x_test_without_lag, y_test_actual)

print("\nOOB Results (With Lagged FedFunds):")
print(f"R²: {rf_oob_with.oob_score_:.4f}")
print(f"MAE: {mean_absolute_error(y_test_actual, rf_oob_with.oob_prediction_):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_actual, rf_oob_with.oob_prediction_)):.4f}")

print("\nOOB Results (Without Lagged FedFunds):")
print(f"R²: {rf_oob_without.oob_score_:.4f}")
print(f"MAE: {mean_absolute_error(y_test_actual, rf_oob_without.oob_prediction_):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_actual, rf_oob_without.oob_prediction_)):.4f}")
