import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

RFModelWithInflationInteraction = joblib.load("rf_model_with_ii.pkl")
RFModelWithoutInflationInteraction = joblib.load("rf_model_without_ii.pkl")

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

startDate = '2005-01-01'
endDate = '2013-12-31'

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

fedFundsTest = fedFundsTest[fedFundsTest['observation_date'] >= "2007-01-01"]
inflationTest = inflationTest[inflationTest['observation_date'] >= "2007-01-01"]
mergedPotGDP = mergedPotGDP[mergedPotGDP['observation_date'] >= "2007-01-01"]
partyData = partyData[partyData['observation_date'] >= "2007-01-01"]

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

featuresWithInflationInteraction = RFModelWithInflationInteraction.feature_names_in_
featuresWithoutInflationInteraction = RFModelWithoutInflationInteraction.feature_names_in_

x_NewWithInflationInteraction = mergedTestData[featuresWithInflationInteraction]
x_NewWithoutInflationInteraction = mergedTestData[featuresWithoutInflationInteraction]

PredsWithInflationInteraction = RFModelWithInflationInteraction.predict(x_NewWithInflationInteraction)
PredsWithoutInflationInteraction = RFModelWithoutInflationInteraction.predict(x_NewWithoutInflationInteraction)

mergedTestData['Predicted_FedFunds_With_II'] = PredsWithInflationInteraction
mergedTestData['Predicted_FedFunds_Without_II'] = PredsWithoutInflationInteraction
mergedTestData[['observation_date', 'Predicted_FedFunds_With_II', 'Predicted_FedFunds_Without_II']].to_csv("Predicted_FedFunds.csv", index=False)

predictedData = pd.read_csv("../Training_and_Testing_Data/Predicted_FedFunds.csv")
actualData = pd.read_csv("../Training_and_Testing_Data/FedfundsTest.csv")
mergedData = predictedData.merge(actualData[['observation_date', 'FEDFUNDS']], on='observation_date', how='inner')

r2WithInflationInteraction = r2_score(mergedData['FEDFUNDS'], mergedData['Predicted_FedFunds_With_II'])
r2WithoutInflationInteraction = r2_score(mergedData['FEDFUNDS'], mergedData['Predicted_FedFunds_Without_II'])

print(f"R² Score (With Inflation Interaction): {r2WithInflationInteraction:.4f}")
print(f"R² Score (Without Inflation Interaction): {r2WithoutInflationInteraction:.4f}")

RFModelWithLag = joblib.load("rf_model_with_lag.pkl")
RFModelWithoutLag = joblib.load("rf_model_without_lag.pkl")

featuresWithLag = ['InflationRate', 'OutputGap', 'FedFundsLag1']
featuresWithoutLag = ['InflationRate', 'OutputGap']

x_testWithLag = mergedTestData[featuresWithLag]
x_testWithoutLag = mergedTestData[featuresWithoutLag]
y_testActual = mergedTestData['FEDFUNDS']

y_predWithLag = RFModelWithLag.predict(x_testWithLag)
y_predWithoutLag = RFModelWithoutLag.predict(x_testWithoutLag)

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
results.append(evaluateRFModel(y_testActual, y_predWithLag, "Random Forest **With** Lagged FedFunds"))
results.append(evaluateRFModel(y_testActual, y_predWithoutLag, "Random Forest **Without** Lagged FedFunds"))

RFTestResults = pd.DataFrame(results)
print(RFTestResults)

def computeVIF(df, features):
    VIFDf = pd.DataFrame()
    VIFDf["Feature"] = features
    VIFDf["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return VIFDf.sort_values(by="VIF", ascending=False)

vifWithLag = computeVIF(mergedTestData, featuresWithLag)
vifWithoutLag = computeVIF(mergedTestData, featuresWithoutLag)

print("\nVIF for With Lagged FedFunds:")
print(vifWithLag)

print("\nVIF for Without Lagged FedFunds:")
print(vifWithoutLag)

pfiWithLag = permutation_importance(RFModelWithLag, x_testWithLag, y_testActual, n_repeats=30, random_state=42)
pfiWithoutLag = permutation_importance(RFModelWithoutLag, x_testWithoutLag, y_testActual, n_repeats=30, random_state=42)

pfi_dfWithLag = pd.DataFrame({
    "Feature": x_testWithLag.columns,
    "Importance": pfiWithLag.importances_mean,
    "StdDev": pfiWithLag.importances_std
}).sort_values(by="Importance", ascending=False)

pfi_dfWithoutLag = pd.DataFrame({
    "Feature": x_testWithoutLag.columns,
    "Importance": pfiWithoutLag.importances_mean,
    "StdDev": pfiWithoutLag.importances_std
}).sort_values(by="Importance", ascending=False)

print("\nPermutation Importance for With Lagged FedFunds:")
print(pfi_dfWithLag)

print("\nPermutation Importance for Without Lagged FedFunds:")
print(pfi_dfWithoutLag)

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

baselineWithr2 = r2_score(y_testActual, y_predWithLag)
baselineWithrmse = np.sqrt(mean_squared_error(y_testActual, y_predWithLag))
baselineWithmae = mean_absolute_error(y_testActual, y_predWithLag)

baselineWithoutr2 = r2_score(y_testActual, y_predWithoutLag)
baselineWithoutrmse = np.sqrt(mean_squared_error(y_testActual, y_predWithoutLag))
baselineWithoutmae = mean_absolute_error(y_testActual, y_predWithoutLag)

looWithLag = LOOAnalysis(x_testWithLag, y_testActual, RFModelWithLag, baselineWithr2, baselineWithrmse, baselineWithmae)
looWithoutLag = LOOAnalysis(x_testWithoutLag, y_testActual, RFModelWithoutLag, baselineWithoutr2, baselineWithoutrmse, baselineWithoutmae)

print("\nLOO Analysis for With Lagged FedFunds:")
print(looWithLag)

print("\nLOO Analysis for Without Lagged FedFunds:")
print(looWithoutLag)

RF_OOBWithLag = RandomForestRegressor(n_estimators=397, max_depth=13, min_samples_leaf=4, min_samples_split=3,oob_score=True, random_state=42)
RF_OOBWithLag.fit(x_testWithLag, y_testActual)

RF_OOBWithoutLag = RandomForestRegressor(n_estimators=397, max_depth=13, min_samples_leaf=4, min_samples_split=3,oob_score=True, random_state=42)
RF_OOBWithoutLag.fit(x_testWithoutLag, y_testActual)

print("\nOOB Results (With Lagged FedFunds):")
print(f"R²: {RF_OOBWithLag.oob_score_:.4f}")
print(f"MAE: {mean_absolute_error(y_testActual, RF_OOBWithLag.oob_prediction_):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_testActual, RF_OOBWithLag.oob_prediction_)):.4f}")

print("\nOOB Results (Without Lagged FedFunds):")
print(f"R²: {RF_OOBWithoutLag.oob_score_:.4f}")
print(f"MAE: {mean_absolute_error(y_testActual, RF_OOBWithoutLag.oob_prediction_):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_testActual, RF_OOBWithoutLag.oob_prediction_)):.4f}")

startDateExtended = '2000-01-01'
endDate = '2013-12-31'

fedFundsExtended = pd.read_csv("../Training_and_Testing_Data/FedfundsTest.csv")
inflationExtended = pd.read_csv("../Training_and_Testing_Data/InflationTest.csv")
realGDPExtended = pd.read_csv("../Training_and_Testing_Data/RealGDPTest.csv")
potGDPExtended = pd.read_csv("../Training_and_Testing_Data/PotGDPTest.csv")

fedFundsExtended['observation_date'] = pd.to_datetime(fedFundsExtended['observation_date'])
inflationExtended['observation_date'] = pd.to_datetime(inflationExtended['observation_date'])
realGDPExtended['observation_date'] = pd.to_datetime(realGDPExtended['observation_date'])
potGDPExtended['observation_date'] = pd.to_datetime(potGDPExtended['observation_date'])

fedFundsExtended = fedFundsExtended[(fedFundsExtended['observation_date'] >= startDateExtended) & (fedFundsExtended['observation_date'] <= endDate)]
inflationExtended = inflationExtended[(inflationExtended['observation_date'] >= startDateExtended) & (inflationExtended['observation_date'] <= endDate)]
realGDPExtended = realGDPExtended[(realGDPExtended['observation_date'] >= startDateExtended) & (realGDPExtended['observation_date'] <= endDate)]
potGDPExtended = potGDPExtended[(potGDPExtended['observation_date'] >= startDateExtended) & (potGDPExtended['observation_date'] <= endDate)]

inflationExtended['InflationRate'] = (inflationExtended['GDPDEF'] / inflationExtended['GDPDEF'].shift(4) - 1) * 100
mergedGap = potGDPExtended[['observation_date', 'GDPPOT']].merge(realGDPExtended[['observation_date', 'GDPC1']], on='observation_date')
mergedGap['OutputGap'] = 100 * (np.log(mergedGap['GDPC1']) - np.log(mergedGap['GDPPOT']))

regressionData = fedFundsExtended[['observation_date', 'FEDFUNDS']].merge(
    inflationExtended[['observation_date', 'InflationRate']], on='observation_date'
).merge(
    mergedGap[['observation_date', 'OutputGap']], on='observation_date'
)

regressionData.dropna(inplace=True)

regressionData.set_index('observation_date', inplace=True)
regressionData.sort_index(inplace=True)

windowSize = 24  
coefs = []
dates = []

for i in range(windowSize, len(regressionData)):
    window = regressionData.iloc[i - windowSize:i]
    X = sm.add_constant(window[['InflationRate', 'OutputGap']])
    y = window['FEDFUNDS']
    model = sm.OLS(y, X).fit()
    coefs.append(model.params)
    dates.append(regressionData.index[i])

coeffDf = pd.DataFrame(coefs, index=dates)

plt.figure(figsize=(10, 6))
plt.plot(coeffDf.index, coeffDf['InflationRate'], label='InflationRate Coef')
plt.plot(coeffDf.index, coeffDf['OutputGap'], label='OutputGap Coef')
plt.axvline(pd.to_datetime('2008-01-01'), color='red', linestyle='--', label='2008')
plt.title('Rolling OLS Coefficients (24-Quarter Window)')
plt.xlabel('Date')
plt.ylabel('Coefficient')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()