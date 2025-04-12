# ======================= Data Processing ======================= #
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from matplotlib import pyplot as plt
import joblib

# Modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# ======================= Load Data ======================= #
fedfundsData = pd.read_csv("FedfundsTotal.csv")
inflationData = pd.read_csv("InflationTotal.csv")
realGDPData = pd.read_csv("RealGDPTotal.csv")
potGDPData = pd.read_csv("PotentialGDPTotal.csv")
partyData = pd.read_csv("PartyTotal.csv")

# Merge all datasets
mergedDataset = (fedfundsData.merge(inflationData, on='observation_date')
                              .merge(realGDPData, on='observation_date')
                              .merge(potGDPData, on='observation_date')
                              .merge(partyData, on='observation_date'))

# ======================= Feature Engineering ======================= #
mergedDataset['observation_date'] = pd.to_datetime(mergedDataset['observation_date'])
mergedDataset['InflationRate'] = ((mergedDataset['GDPDEF'] / mergedDataset['GDPDEF'].shift(4) - 1) * 100)
mergedDataset['OutputGap'] = 100 * (np.log(mergedDataset['GDPC1']) - np.log(mergedDataset['GDPPOT']))
mergedDataset['FedFundsLag1'] = mergedDataset['FEDFUNDS'].shift(1)
mergedDataset['InflationInteraction'] = mergedDataset['InflationRate'] * mergedDataset['PresidentParty']
mergedDataset['OutputGapInteraction'] = mergedDataset['OutputGap'] * mergedDataset['PresidentParty']

# Drop unnecessary columns
mergedDataset.drop(columns=['GDPDEF', 'GDPPOT', 'GDPC1'], inplace=True)
mergedDataset.dropna(inplace=True)

# ======================= Model Preparation ======================= #
features = ['InflationRate', 'OutputGap', 'PresidentParty',
            'InflationInteraction', 'OutputGapInteraction', 'FedFundsLag1']
x = mergedDataset[features]
y = mergedDataset['FEDFUNDS']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)


# ======================= Baseline Random Forest Model ======================= #
rfRegressor = RandomForestRegressor(
    max_depth=13,
    min_samples_leaf=4,
    min_samples_split=3,
    n_estimators=397,
    random_state=42
)
rfRegressor.fit(xTrain, yTrain)

# Predictions & Performance
yPred = rfRegressor.predict(xTest)
baseline_mae = mean_absolute_error(yTest, yPred)
baseline_rmse = np.sqrt(mean_squared_error(yTest, yPred))
baseline_r2 = r2_score(yTest, yPred)

# Display Performance
print(f"Mean Absolute Error (MAE): {baseline_mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {baseline_rmse:.4f}")
print(f"R² Score: {baseline_r2:.4f}")


# ======================= Permutation Feature Importance ======================= #
result = permutation_importance(rfRegressor, xTest, yTest, n_repeats=30, random_state=42, scoring='r2')

importances_df = pd.DataFrame({
    'Variable': xTest.columns,
    'Importance': result.importances_mean,
    'StdDev': result.importances_std
}).sort_values(by='Importance', ascending=False)

print("\nPermutation Feature Importance:")
print(importances_df)

# Plot Permutation Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(importances_df['Variable'], importances_df['Importance'], xerr=importances_df['StdDev'], color='skyblue')
plt.xlabel('Importance (Decrease in R²)')
plt.ylabel('Feature')
plt.title('Permutation Feature Importance')
plt.show()


# ======================= Leave-One-Out Analysis (LOO) ======================= #
results = []
for feature in features:
    # Drop One Feature
    xTrainDrop = xTrain.drop(columns=[feature])
    xTestDrop = xTest.drop(columns=[feature])

    # Train Model Without This Feature
    rf = RandomForestRegressor(n_estimators=397, max_depth=13,
                               min_samples_leaf=4, min_samples_split=3, random_state=42)
    rf.fit(xTrainDrop, yTrain)

    # Predict & Evaluate
    yPred = rf.predict(xTestDrop)
    mae = mean_absolute_error(yTest, yPred)
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    r2 = r2_score(yTest, yPred)

    # Performance Drop Compared to Baseline
    mae_drop = mae - baseline_mae
    rmse_drop = rmse - baseline_rmse
    r2_drop = baseline_r2 - r2

    results.append({
        'Feature': feature,
        'MAE Drop': mae_drop,
        'RMSE Drop': rmse_drop,
        'R² Drop': r2_drop
    })

# Display LOO Results
loo_results = pd.DataFrame(results).sort_values(by='R² Drop', ascending=False)
print("\nLeave-One-Out Analysis:")
print(loo_results)


def noise_sensitivity_analysis(x_train, x_test, y_train, y_test, noise_levels=[0.01, 0.05, 0.1, 0.2]):
    results = []
    rf = RandomForestRegressor(n_estimators=397, max_depth=13, min_samples_leaf=4, min_samples_split=3, random_state=42)
    rf.fit(x_train, y_train)

    baseline_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(x_test)))

    for noise in noise_levels:
        x_test_noisy = x_test + np.random.normal(loc=0, scale=noise, size=x_test.shape)

        y_pred = rf.predict(x_test_noisy)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_increase = ((rmse - baseline_rmse) / baseline_rmse) * 100  # Percentage increase

        results.append({
            'Noise Level': noise,
            'RMSE': rmse,
            'RMSE Increase (%)': rmse_increase
        })

    return pd.DataFrame(results)


# Run analysis
noise_results = noise_sensitivity_analysis(xTrain, xTest, yTrain, yTest)
print(noise_results)
'''
Unexpectedly, performance improved slightly with low noise, suggesting possible overfitting to the original data. Small noise may act as regularization, improving generalization.
Higher noise starts to degrade performance, showing the model is sensitive to moderate disturbances.
Performance drop is smaller than at 0.10, suggesting the model may be less sensitive to larger noise levels due to random forest ensemble effects.
'''
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(vif_data.sort_values(by="VIF", ascending=False))
'''
VIF (Variance Inflation Factor) measures how much the variance of a regression coefficient is inflated due to multicollinearity with other predictors; a VIF > 5 indicates high multicollinearity.
'''

# ======================= Model Preparation- Without Inflation Interaction ======================= #
features2 = ['InflationRate', 'OutputGap', 'PresidentParty', 'OutputGapInteraction', 'FedFundsLag1']
x2 = mergedDataset[features2]
y2 = mergedDataset['FEDFUNDS']

xTrain2, xTest2, yTrain2, yTest2 = train_test_split(x2, y2, test_size=0.2, random_state=42, shuffle=False)


# ======================= Baseline Random Forest Model- Without Inflation Interaction ======================= #
rfRegressor2 = RandomForestRegressor(
    max_depth=13,
    min_samples_leaf=4,
    min_samples_split=3,
    n_estimators=397,
    random_state=42
)
rfRegressor2.fit(xTrain2, yTrain2)

# Predictions & Performance
yPred2 = rfRegressor2.predict(xTest2)
baseline_mae2 = mean_absolute_error(yTest2, yPred2)
baseline_rmse2 = np.sqrt(mean_squared_error(yTest2, yPred2))
baseline_r22 = r2_score(yTest2, yPred2)

# Display Performance
print(f"Mean Absolute Error (MAE): {baseline_mae2:.4f}")
print(f"Root Mean Squared Error (RMSE): {baseline_rmse2:.4f}")
print(f"R² Score: {baseline_r22:.4f}")


# ======================= Permutation Feature Importance- Without II ======================= #
result2 = permutation_importance(rfRegressor2, xTest2, yTest2, n_repeats=30, random_state=42, scoring='r2')

importances_df2 = pd.DataFrame({
    'Variable': xTest2.columns,
    'Importance': result2.importances_mean,
    'StdDev': result2.importances_std
}).sort_values(by='Importance', ascending=False)

print("\nPermutation Feature Importance:")
print(importances_df2)

# Plot Permutation Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(importances_df2['Variable'], importances_df2['Importance'], xerr=importances_df2['StdDev'], color='skyblue')
plt.xlabel('Importance (Decrease in R²) Without II')
plt.ylabel('Feature Without II')
plt.title('Permutation Feature Importance Without II')
plt.show()

# ======================= Leave-One-Out Analysis (LOO)- Without II ======================= #
results2 = []
for feature in features2:
    # Drop One Feature
    xTrainDrop2 = xTrain2.drop(columns=[feature])
    xTestDrop2 = xTest2.drop(columns=[feature])

    # Train Model Without This Feature
    rf = RandomForestRegressor(n_estimators=397, max_depth=13,
                               min_samples_leaf=4, min_samples_split=3, random_state=42)
    rf.fit(xTrainDrop2, yTrain2)

    # Predict & Evaluate
    yPred2 = rf.predict(xTestDrop2)
    mae2 = mean_absolute_error(yTest2, yPred2)
    rmse2 = np.sqrt(mean_squared_error(yTest2, yPred2))
    r22 = r2_score(yTest2, yPred2)

    # Performance Drop Compared to Baseline
    mae_drop2 = mae2 - baseline_mae
    rmse_drop2 = rmse2 - baseline_rmse
    r2_drop2 = baseline_r2 - r22

    results2.append({
        'Feature': feature,
        'MAE Drop': mae_drop2,
        'RMSE Drop': rmse_drop2,
        'R² Drop': r2_drop2
    })

# Display LOO Results
loo_results2 = pd.DataFrame(results2).sort_values(by='R² Drop', ascending=False)
print("\nLeave-One-Out Analysis:")
print(loo_results2)

# Run analysis Without II (Noise)
noise_results2 = noise_sensitivity_analysis(xTrain2, xTest2, yTrain2, yTest2)
print(noise_results2)

# Calculate VIF for each predictor Without II
vif_data2 = pd.DataFrame()
vif_data2["Feature"] = x2.columns
vif_data2["VIF"] = [variance_inflation_factor(x2.values, i) for i in range(x2.shape[1])]
print(vif_data2.sort_values(by="VIF", ascending=False))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# Initialize the Random Forest Regressor with OOB enabled
rfRegressor = RandomForestRegressor(
    n_estimators=397,  # Use optimal number of trees from previous tuning
    max_depth=13,
    min_samples_leaf=4,
    min_samples_split=3,
    oob_score=True,  # Enable OOB testing
    random_state=42
)

# Fit the model on the entire dataset
rfRegressor.fit(x2, y2)

# OOB predictions and performance
oob_r2 = rfRegressor.oob_score_
oob_predictions = rfRegressor.oob_prediction_

# Calculate additional metrics using OOB predictions
oob_mae = mean_absolute_error(y2, oob_predictions)
oob_rmse = np.sqrt(mean_squared_error(y2, oob_predictions))

# Print results
print(f"Out-of-Bag R²: {oob_r2:.4f}")
print(f"Out-of-Bag MAE: {oob_mae:.4f}")
print(f"Out-of-Bag RMSE: {oob_rmse:.4f}")

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_score
import numpy as np
import pandas as pd

loo = LeaveOneOut()
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=13,
    min_samples_leaf=4,
    min_samples_split=3,
    random_state=42
)

mse_values = []

for train_index, test_index in loo.split(x2):
    xTrain, xTest = x2.iloc[train_index], x2.iloc[test_index]
    yTrain, yTest = y2.iloc[train_index], y2.iloc[test_index]

    rf.fit(xTrain, yTrain)
    yPred = rf.predict(xTest)
    mse_values.append(mean_squared_error(yTest, yPred))

# Final LOOCV Error
print(f"LOOCV Mean Squared Error: {np.mean(mse_values):.4f}")
print(f"LOOCV RMSE: {np.sqrt(np.mean(mse_values)):.4f}")

'''
1. Permutation Feature Importance

    Process: Randomly shuffle the values of one variable while keeping all others unchanged.
    Effect: Measures how much the model’s performance (e.g., R² or RMSE) drops due to the disruption.
    Interpretation: A larger drop means the variable is crucial; a small or negative drop means it's less helpful or even harmful.
    Key Insight: This shows the marginal impact of a variable while accounting for interactions with other predictors.'''


'''
2. Leave-One-Out Analysis (LOO or Drop-One-Out)

    Process: Completely remove one variable from the dataset and re-train the model from scratch.
    Effect: Compare the model’s performance without that variable against the baseline with all variables.
    Interpretation: A large performance drop indicates that the variable is essential. A small drop or performance improvement means the variable isn’t adding value.
🔎 Key Insight: This shows the independent contribution of a variable, isolated from other predictors.
'''

'''
Output Gap does have rpedictive reelvance but may be introducing noise due to multicollinearity or interaction w other variables'''


#out of bag error: 1/3 of trees will be trained without obs 1, output avg across 3 and

#Save Models
joblib.dump(rfRegressor, "rf_model_with_ii.pkl")
joblib.dump(rfRegressor2, "rf_model_without_ii.pkl")

# ======================= Model With Lagged FedFunds ======================= #
features_with_lag = ['InflationRate', 'OutputGap', 'FedFundsLag1']
x_with_lag = mergedDataset[features_with_lag].copy()
y_with_lag = mergedDataset['FEDFUNDS']

rf_with_lag = RandomForestRegressor(
    n_estimators=397,
    max_depth=13,
    min_samples_leaf=4,
    min_samples_split=3,
    random_state=42
)
rf_with_lag.fit(x_with_lag, y_with_lag)
joblib.dump(rf_with_lag, "rf_model_with_lag.pkl")

# ======================= Model Without Lagged FedFunds ======================= #
features_without_lag = ['InflationRate', 'OutputGap']
x_without_lag = mergedDataset[features_without_lag].copy()
y_without_lag = mergedDataset['FEDFUNDS']

rf_without_lag = RandomForestRegressor(
    n_estimators=397,
    max_depth=13,
    min_samples_leaf=4,
    min_samples_split=3,
    random_state=42
)
rf_without_lag.fit(x_without_lag, y_without_lag)
joblib.dump(rf_without_lag, "rf_model_without_lag.pkl")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def calculate_rf_metrics(model, X_train, y_train, X_test, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # In-sample
    r2_in = r2_score(y_train, y_train_pred)
    rmse_in = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_in = mean_absolute_error(y_train, y_train_pred)
    mse_mean_in = mean_squared_error(y_train, y_train_pred) / y_train.mean()
    adj_r2_in = 1 - (1 - r2_in) * ((len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))

    # Out-of-sample
    r2_out = r2_score(y_test, y_test_pred)
    rmse_out = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_out = mean_absolute_error(y_test, y_test_pred)
    mse_mean_out = mean_squared_error(y_test, y_test_pred) / y_test.mean()

    return {
        "Model": model_name,
        "In-sample R-squared": r2_in,
        "In-sample RMSE": rmse_in,
        "In-sample MAE": mae_in,
        "Out-of-sample RMSE": rmse_out,
        "Out-of-sample MAE": mae_out,
        "Adjusted R-squared": adj_r2_in,
        "MSE/Mean (In-sample)": mse_mean_in,
        "MSE/Mean (Out-of-sample)": mse_mean_out
    }

from sklearn.model_selection import train_test_split

# Split for "With Lagged FedFunds"
xTrain_lag, xTest_lag, yTrain_lag, yTest_lag = train_test_split(
    x_with_lag, y_with_lag, test_size=0.2, shuffle=False, random_state=42)

# Split for "Without Lagged FedFunds"
xTrain_nolag, xTest_nolag, yTrain_nolag, yTest_nolag = train_test_split(
    x_without_lag, y_without_lag, test_size=0.2, shuffle=False, random_state=42)

results = []

results.append(calculate_rf_metrics(
    rf_with_lag, xTrain_lag, yTrain_lag, xTest_lag, yTest_lag,
    "Random Forest **With** Lagged FedFunds"
))

results.append(calculate_rf_metrics(
    rf_without_lag, xTrain_nolag, yTrain_nolag, xTest_nolag, yTest_nolag,
    "Random Forest **Without** Lagged FedFunds"
))

metrics_df = pd.DataFrame(results)
print(metrics_df)

from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(X, label):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(f"\nVIF for {label}:\n", vif_data.sort_values(by="VIF", ascending=False))

# Compute for both models
compute_vif(x_with_lag, "With Lagged FedFunds")
compute_vif(x_without_lag, "Without Lagged FedFunds")

from sklearn.inspection import permutation_importance


def perm_importance(model, X_test, y_test, label):
    result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring='r2')
    importances_df = pd.DataFrame({
        'Variable': X_test.columns,
        'Importance': result.importances_mean,
        'StdDev': result.importances_std
    }).sort_values(by='Importance', ascending=False)

    print(f"\nPermutation Importance for {label}:\n", importances_df)


# Run for both
perm_importance(rf_with_lag, xTest_lag, yTest_lag, "With Lagged FedFunds")
perm_importance(rf_without_lag, xTest_nolag, yTest_nolag, "Without Lagged FedFunds")

def loo_analysis(model, X_train, y_train, X_test, y_test, label):
    results = []
    baseline_r2 = r2_score(y_test, model.predict(X_test))
    baseline_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    baseline_mae = mean_absolute_error(y_test, model.predict(X_test))

    for feature in X_train.columns:
        X_train_drop = X_train.drop(columns=[feature])
        X_test_drop = X_test.drop(columns=[feature])

        rf = RandomForestRegressor(n_estimators=397, max_depth=13,
                                   min_samples_leaf=4, min_samples_split=3, random_state=42)
        rf.fit(X_train_drop, y_train)
        y_pred = rf.predict(X_test_drop)

        r2_drop = baseline_r2 - r2_score(y_test, y_pred)
        rmse_drop = np.sqrt(mean_squared_error(y_test, y_pred)) - baseline_rmse
        mae_drop = mean_absolute_error(y_test, y_pred) - baseline_mae

        results.append({
            "Feature": feature,
            "R² Drop": r2_drop,
            "RMSE Drop": rmse_drop,
            "MAE Drop": mae_drop
        })

    df = pd.DataFrame(results).sort_values(by='R² Drop', ascending=False)
    print(f"\nLOO Analysis for {label}:\n", df)

# Run for both
loo_analysis(rf_with_lag, xTrain_lag, yTrain_lag, xTest_lag, yTest_lag, "With Lagged FedFunds")
loo_analysis(rf_without_lag, xTrain_nolag, yTrain_nolag, xTest_nolag, yTest_nolag, "Without Lagged FedFunds")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ========== Model With Lagged FedFunds (OOB) ==========
rf_with_lag_oob = RandomForestRegressor(
    n_estimators=397,
    max_depth=13,
    min_samples_leaf=4,
    min_samples_split=3,
    oob_score=True,
    bootstrap=True,
    random_state=42
)
rf_with_lag_oob.fit(x_with_lag, y_with_lag)

oob_r2_with = rf_with_lag_oob.oob_score_
oob_pred_with = rf_with_lag_oob.oob_prediction_
oob_mae_with = mean_absolute_error(y_with_lag, oob_pred_with)
oob_rmse_with = np.sqrt(mean_squared_error(y_with_lag, oob_pred_with))

print("OOB Results (With Lagged FedFunds):")
print(f"R²: {oob_r2_with:.4f}")
print(f"MAE: {oob_mae_with:.4f}")
print(f"RMSE: {oob_rmse_with:.4f}")

# ========== Model Without Lagged FedFunds (OOB) ==========
rf_without_lag_oob = RandomForestRegressor(
    n_estimators=397,
    max_depth=13,
    min_samples_leaf=4,
    min_samples_split=3,
    oob_score=True,
    bootstrap=True,
    random_state=42
)
rf_without_lag_oob.fit(x_without_lag, y_without_lag)

oob_r2_without = rf_without_lag_oob.oob_score_
oob_pred_without = rf_without_lag_oob.oob_prediction_
oob_mae_without = mean_absolute_error(y_without_lag, oob_pred_without)
oob_rmse_without = np.sqrt(mean_squared_error(y_without_lag, oob_pred_without))

print("\nOOB Results (Without Lagged FedFunds):")
print(f"R²: {oob_r2_without:.4f}")
print(f"MAE: {oob_mae_without:.4f}")
print(f"RMSE: {oob_rmse_without:.4f}")
