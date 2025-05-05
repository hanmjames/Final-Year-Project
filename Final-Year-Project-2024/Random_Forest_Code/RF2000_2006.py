import joblib
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor

RF_ModelWithInflationInteraction = joblib.load("rf_model_with_ii.pkl")
RF_ModelWithoutInflationInteraction = joblib.load("rf_model_without_ii.pkl")

fedFundsTest = pd.read_csv("../Training_and_Testing_Data/FedfundsTest.csv")
inflationTest = pd.read_csv("../Training_and_Testing_Data/InflationTest.csv")
realGDPTest = pd.read_csv("../Training_and_Testing_Data/RealGDPTest.csv")
potGDPTest = pd.read_csv("../Training_and_Testing_Data/PotGDPTest.csv")
partyData = pd.read_csv("../Training_and_Testing_Data/partyTotal.csv")

fedFundsTest['observation_date'] = pd.to_datetime(fedFundsTest['observation_date'])
inflationTest['observation_date'] = pd.to_datetime(inflationTest['observation_date'])
realGDPTest['observation_date'] = pd.to_datetime(realGDPTest['observation_date'])
potGDPTest['observation_date'] = pd.to_datetime(potGDPTest['observation_date'])
partyData['observation_date'] = pd.to_datetime(partyData['observation_date'], errors='coerce', dayfirst=True)

startDate = '1998-01-01'
endDate = '2006-12-31'

fedFundsTest = fedFundsTest[(fedFundsTest['observation_date'] >= startDate) & (fedFundsTest['observation_date'] <= endDate)]
inflationTest = inflationTest[(inflationTest['observation_date'] >= startDate) & (inflationTest['observation_date'] <= endDate)]
realGDPTest = realGDPTest[(realGDPTest['observation_date'] >= startDate) & (realGDPTest['observation_date'] <= endDate)]
potGDPTest = potGDPTest[(potGDPTest['observation_date'] >= startDate) & (potGDPTest['observation_date'] <= endDate)]
partyData = partyData[(partyData['observation_date'] >= startDate) & (partyData['observation_date'] <= endDate)]

inflationTest['Inflation_Rate'] = (
    (inflationTest['GDPDEF'] / inflationTest['GDPDEF'].shift(4) - 1) * 100
)

mergedPotGDP = potGDPTest[['observation_date', 'GDPPOT']].merge(
    realGDPTest[['observation_date', 'GDPC1']], on='observation_date'
)
mergedPotGDP['OutputGap'] = 100 * (
    (np.log(mergedPotGDP['GDPC1']) - np.log(mergedPotGDP['GDPPOT']))
)

for lag in range(1, 4):
    fedFundsTest[f'FEDFUNDS_Lag{lag}'] = fedFundsTest['FEDFUNDS'].shift(lag)
    inflationTest[f'Inflation_Rate_Lag{lag}'] = inflationTest['Inflation_Rate'].shift(lag)
    mergedPotGDP[f'OutputGap_Lag{lag}'] = mergedPotGDP['OutputGap'].shift(lag)

fedFundsTest = fedFundsTest[fedFundsTest['observation_date'] >= "2000-01-01"]
inflationTest = inflationTest[inflationTest['observation_date'] >= "2000-01-01"]
mergedPotGDP = mergedPotGDP[mergedPotGDP['observation_date'] >= "2000-01-01"]
partyData = partyData[partyData['observation_date'] >= "2000-01-01"]

mergedTestData = pd.concat(
    [
        fedFundsTest.set_index("observation_date"),
        inflationTest.set_index("observation_date"),
        mergedPotGDP.set_index("observation_date"),
        partyData.set_index("observation_date")
    ],
    axis=1
).reset_index()

mergedTestData['InflationInteraction'] = mergedTestData['Inflation_Rate'] * mergedTestData['PresidentParty']
mergedTestData['OutputGapInteraction'] = mergedTestData['OutputGap'] * mergedTestData['PresidentParty']
mergedTestData.rename(columns={'FEDFUNDS_Lag1': 'FedFundsLag1', 'Inflation_Rate': 'InflationRate'}, inplace=True)

featuresWithInflationInteraction = RF_ModelWithInflationInteraction.feature_names_in_
featuresWithoutInflationInteraction = RF_ModelWithoutInflationInteraction.feature_names_in_
x_newWithInflationInteraction = mergedTestData[featuresWithInflationInteraction]
x_newWithoutInflationInteraction = mergedTestData[featuresWithoutInflationInteraction]
predictionsWithInflationInteraction = RF_ModelWithInflationInteraction.predict(x_newWithInflationInteraction)
predictionsWithoutInflationInteraction = RF_ModelWithoutInflationInteraction.predict(x_newWithoutInflationInteraction)

mergedTestData['Predicted_FedFundsWithInflationInteraction'] = predictionsWithInflationInteraction
mergedTestData['Predicted_FedFundsWithoutInflationInteraction'] = predictionsWithoutInflationInteraction
mergedTestData[['observation_date', 'Predicted_FedFundsWithInflationInteraction', 'Predicted_FedFundsWithoutInflationInteraction']].to_csv("Predicted_FedFunds.csv", index=False)

predictedData = pd.read_csv("../Training_and_Testing_Data/Predicted_FedFunds.csv")
actualData = pd.read_csv("../Training_and_Testing_Data/FedfundsTest.csv")
mergedData = predictedData.merge(actualData[['observation_date', 'FEDFUNDS']], on='observation_date', how='inner')

r2WithInflationInteraction = r2_score(mergedData['FEDFUNDS'], mergedData['Predicted_FedFunds_With_II'])
r2WithoutInflationInteraction = r2_score(mergedData['FEDFUNDS'], mergedData['Predicted_FedFunds_Without_II'])

print(f"R² Score (With Inflation Interaction): {r2WithInflationInteraction:.4f}")
print(f"R² Score (Without Inflation Interaction): {r2WithoutInflationInteraction:.4f}")

rf_modelWithLag = joblib.load("rf_model_with_lag.pkl")
rf_modelWithoutLag = joblib.load("rf_model_without_lag.pkl")

featuresWithLag = ['InflationRate', 'OutputGap', 'FedFundsLag1']
featuresWithoutLag = ['InflationRate', 'OutputGap']
x_testWithLag = mergedTestData[featuresWithLag]
x_testWithoutLag = mergedTestData[featuresWithoutLag]
y_test_actual = mergedTestData['FEDFUNDS']
y_predWithLag = rf_modelWithLag.predict(x_testWithLag)
y_predWithoutLag = rf_modelWithoutLag.predict(x_testWithoutLag)

def evaluateRFModel(y_true, y_pred, model_name):
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

results = []
results.append(evaluateRFModel(y_test_actual, y_predWithLag, "Random Forest **With** Lagged FedFunds"))
results.append(evaluateRFModel(y_test_actual, y_predWithoutLag, "Random Forest **Without** Lagged FedFunds"))

rf_test_results = pd.DataFrame(results)
print(rf_test_results)

def computeVIF(df, features):
    vif_df = pd.DataFrame()
    vif_df["Feature"] = features
    vif_df["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_df.sort_values(by="VIF", ascending=False)

VIFWithLag = computeVIF(mergedTestData, featuresWithLag)
VIFWithoutLag = computeVIF(mergedTestData, featuresWithoutLag)

print("\nVIF for With Lagged FedFunds:")
print(VIFWithLag)

print("\nVIF for Without Lagged FedFunds:")
print(VIFWithoutLag)

PFIWithLag = permutation_importance(rf_modelWithLag, x_testWithLag, y_test_actual, n_repeats=30, random_state=42)
PFIWithoutLag = permutation_importance(rf_modelWithoutLag, x_testWithoutLag, y_test_actual, n_repeats=30, random_state=42)

PFIWithLagDf = pd.DataFrame({
    "Feature": x_testWithLag.columns,
    "Importance": PFIWithLag.importances_mean,
    "StdDev": PFIWithLag.importances_std
}).sort_values(by="Importance", ascending=False)

PFIWithoutLagDf = pd.DataFrame({
    "Feature": x_testWithoutLag.columns,
    "Importance": PFIWithoutLag.importances_mean,
    "StdDev": PFIWithoutLag.importances_std
}).sort_values(by="Importance", ascending=False)

print("\nPermutation Importance for With Lagged FedFunds:")
print(PFIWithLagDf)

print("\nPermutation Importance for Without Lagged FedFunds:")
print(PFIWithoutLagDf)

def LOOAnalysis(x_test, y_test, model_base, baseline_r2, baseline_rmse, baseline_mae):
    results = []
    for feature in x_test.columns:
        x_testDrop = x_test.drop(columns=[feature])
        model = RandomForestRegressor(n_estimators=397,max_depth=13,min_samples_leaf=4,min_samples_split=3,random_state=42)
        model.fit(x_testDrop, y_test)
        y_predDrop = model.predict(x_testDrop)

        r2Drop = baseline_r2 - r2_score(y_test, y_predDrop)
        rmseDrop = np.sqrt(mean_squared_error(y_test, y_predDrop)) - baseline_rmse
        maeDrop = mean_absolute_error(y_test, y_predDrop) - baseline_mae

        results.append({
            'Feature': feature,
            'R² Drop': round(r2Drop, 2),
            'RMSE Drop': round(rmseDrop, 2),
            'MAE Drop': round(maeDrop, 2)
        })

    return pd.DataFrame(results).sort_values(by='R² Drop', ascending=False)

baselineWithr2 = r2_score(y_test_actual, y_predWithLag)
baselineWithrmse = np.sqrt(mean_squared_error(y_test_actual, y_predWithLag))
baselineWithmae = mean_absolute_error(y_test_actual, y_predWithLag)

baselineWithoutr2 = r2_score(y_test_actual, y_predWithoutLag)
baselineWithoutrmse = np.sqrt(mean_squared_error(y_test_actual, y_predWithoutLag))
baselineWithoutmae = mean_absolute_error(y_test_actual, y_predWithoutLag)

looWithLag = LOOAnalysis(x_testWithLag, y_test_actual, rf_modelWithLag, baselineWithr2, baselineWithrmse, baselineWithmae)
looWithoutLag = LOOAnalysis(x_testWithoutLag, y_test_actual, rf_modelWithoutLag, baselineWithoutr2, baselineWithoutrmse, baselineWithoutmae)

print("\nLOO Analysis for With Lagged FedFunds:")
print(looWithLag)

print("\nLOO Analysis for Without Lagged FedFunds:")
print(looWithoutLag)

RF_OOBWith = RandomForestRegressor(n_estimators=397, max_depth=13, min_samples_leaf=4, min_samples_split=3,oob_score=True, random_state=42)
RF_OOBWith.fit(x_testWithLag, y_test_actual)

RF_OOBWithout = RandomForestRegressor(n_estimators=397, max_depth=13, min_samples_leaf=4, min_samples_split=3,oob_score=True, random_state=42)
RF_OOBWithout.fit(x_testWithoutLag, y_test_actual)

print("\nOOB Results (With Lagged FedFunds):")
print(f"R²: {RF_OOBWith.oob_score_:.4f}")
print(f"MAE: {mean_absolute_error(y_test_actual, RF_OOBWith.oob_prediction_):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_actual, RF_OOBWith.oob_prediction_)):.4f}")

print("\nOOB Results (Without Lagged FedFunds):")
print(f"R²: {RF_OOBWithout.oob_score_:.4f}")
print(f"MAE: {mean_absolute_error(y_test_actual, RF_OOBWithout.oob_prediction_):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_actual, RF_OOBWithout.oob_prediction_)):.4f}")