import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.api import add_constant
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

from Model_Training.RegressionHouses import ols_2002WithLagQ1, ols_2002WithoutLagQ1, ols_2002WithoutLagAll, ols_1997WithoutLagAll, ols_1997WithoutLagQ1, ols_1997WithLagAll, ols_1997WithLagQ1, ols_2002WithLagAll, results_1997WithoutLaggedQ1, results_1997_withLaggedQ1, results_2002_withLaggedQ1, results_1997WithoutLaggedAll, results_2002_withLaggedAll, results_2002WithoutLaggedQ1, results_1997_withLaggedAll, results_2002WithoutLaggedAll

fedFundsTest = pd.read_csv("../Training_and_Testing_Data/FedfundsTest.csv")
inflationTest = pd.read_csv("../Training_and_Testing_Data/InflationTest.csv")
realGDPTest = pd.read_csv("../Training_and_Testing_Data/RealGDPTest.csv")
potGDPTest = pd.read_csv("../Training_and_Testing_Data/PotGDPTest.csv")
housingTest = pd.read_csv("../Training_and_Testing_Data/HousingTest.csv")
oilTest = pd.read_csv("../Training_and_Testing_Data/OilPriceTest.csv")
unemploymentTest = pd.read_csv("../Training_and_Testing_Data/UnemploymentTest.csv")

fedFundsTest['observation_date'] = pd.to_datetime(fedFundsTest['observation_date'])
inflationTest['observation_date'] = pd.to_datetime(inflationTest['observation_date'])
realGDPTest['observation_date'] = pd.to_datetime(realGDPTest['observation_date'])
potGDPTest['observation_date'] = pd.to_datetime(potGDPTest['observation_date'])
housingTest['observation_date'] = pd.to_datetime(housingTest['observation_date'])
oilTest['observation_date'] = pd.to_datetime(oilTest['observation_date'])
unemploymentTest['observation_date'] = pd.to_datetime(unemploymentTest['observation_date'])

startDate = '2005-01-01'
endDate = '2013-12-31'

fedFundsTest = fedFundsTest[(fedFundsTest['observation_date'] >= startDate) & (fedFundsTest['observation_date'] <= endDate)]
inflationTest = inflationTest[(inflationTest['observation_date'] >= startDate) & (inflationTest['observation_date'] <= endDate)]
realGDPTest = realGDPTest[(realGDPTest['observation_date'] >= startDate) & (realGDPTest['observation_date'] <= endDate)]
potGDPTest = potGDPTest[(potGDPTest['observation_date'] >= startDate) & (potGDPTest['observation_date'] <= endDate)]
housingTest = housingTest[(housingTest['observation_date'] >= startDate) & (housingTest['observation_date'] <= endDate)]
oilTest = oilTest[(oilTest['observation_date'] >= startDate) & (oilTest['observation_date'] <= endDate)]
unemploymentTest = unemploymentTest[(unemploymentTest['observation_date'] >= startDate) & (unemploymentTest['observation_date'] <= endDate)]

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
    housingTest[f'Housing_Lag{lag}'] = housingTest['HOUST'].shift(lag)

fedFundsTest = fedFundsTest[fedFundsTest['observation_date'] >= "2000-01-01"]
inflationTest = inflationTest[inflationTest['observation_date'] >= "2000-01-01"]
mergedPotGDP = mergedPotGDP[mergedPotGDP['observation_date'] >= "2000-01-01"]
housingTest = housingTest[housingTest['observation_date'] >= "2000-01-01"]
oilTest = oilTest[oilTest['observation_date'] >= "2000-01-01"]
unemploymentTest = unemploymentTest[unemploymentTest['observation_date'] >= "2000-01-01"]

mergedTestData = (
    fedFundsTest[['observation_date', 'FEDFUNDS', 'FEDFUNDS_Lag1', 'FEDFUNDS_Lag2', 'FEDFUNDS_Lag3']]
    .merge(inflationTest[['observation_date', 'Inflation_Rate', 'Inflation_Rate_Lag1', 'Inflation_Rate_Lag2', 'Inflation_Rate_Lag3']], on='observation_date')
    .merge(mergedPotGDP[['observation_date', 'OutputGap', 'OutputGap_Lag1', 'OutputGap_Lag2', 'OutputGap_Lag3']], on='observation_date')
    .merge(housingTest[['observation_date', 'HOUST', 'Housing_Lag1', 'Housing_Lag2', 'Housing_Lag3']], on='observation_date')

)

mergedTestData.rename(columns={
    "FEDFUNDS": "FEDFUNDS_19970107",
    "FEDFUNDS_Lag1": "FedFunds_1997_Lag1",
    "Inflation_Rate": "Inflation_Rate_1997",
    "Inflation_Rate_Lag1": "Inflation_Rate_1997_Lag1",
    "Inflation_Rate_Lag2": "Inflation_Rate_1997_Lag2",
    "Inflation_Rate_Lag3": "Inflation_Rate_1997_Lag3",
    "OutputGap": "OutputGap_1997",
    "OutputGap_Lag1": "OutputGap_1997_Lag1",
    "OutputGap_Lag2": "OutputGap_1997_Lag2",
    "OutputGap_Lag3": "OutputGap_1997_Lag3",
    "HOUST": "HOUST_19970122",
    "Housing_Lag1": "Houses_1997_Lag1",
    "Housing_Lag2": "Houses_1997_Lag2",
    "Housing_Lag3": "Houses_1997_Lag3"
}, inplace=True)

mergedTestData["FEDFUNDS_20020108"] = mergedTestData["FEDFUNDS_19970107"]
mergedTestData["FedFunds_2002_Lag1"] = mergedTestData["FedFunds_1997_Lag1"]
mergedTestData["Inflation_Rate_2002"] = mergedTestData["Inflation_Rate_1997"]
mergedTestData["Inflation_Rate_2002_Lag1"] = mergedTestData["Inflation_Rate_1997_Lag1"]
mergedTestData["Inflation_Rate_2002_Lag2"] = mergedTestData["Inflation_Rate_1997_Lag2"]
mergedTestData["Inflation_Rate_2002_Lag3"] = mergedTestData["Inflation_Rate_1997_Lag3"]
mergedTestData["OutputGap_2002"] = mergedTestData["OutputGap_1997"]
mergedTestData["OutputGap_2002_Lag1"] = mergedTestData["OutputGap_1997_Lag1"]
mergedTestData["OutputGap_2002_Lag2"] = mergedTestData["OutputGap_1997_Lag2"]
mergedTestData["OutputGap_2002_Lag3"] = mergedTestData["OutputGap_1997_Lag3"]
mergedTestData["HOUST_20020117"] = mergedTestData["HOUST_19970122"]
mergedTestData["Houses_2002_Lag1"] = mergedTestData["Houses_1997_Lag1"]
mergedTestData["Houses_2002_Lag2"] = mergedTestData["Houses_1997_Lag2"]
mergedTestData["Houses_2002_Lag3"] = mergedTestData["Houses_1997_Lag3"]

scaler = joblib.load("../Model_Training/HousesScaler.joblib")
standardizeCols = joblib.load("../Model_Training/HousesScalerColumns.joblib")
mergedTestData = mergedTestData.dropna(subset=standardizeCols)
mergedTestData = mergedTestData[standardizeCols + ['observation_date']]
mergedTestData[standardizeCols] = scaler.transform(mergedTestData[standardizeCols])
mergedTestData = mergedTestData[mergedTestData['observation_date'] >= "2007-01-01"]

olsModels = [
    (
        ols_1997WithoutLagQ1,
        "OLS 1997 Without Lagged FEDFUNDS (Q1)",
        ["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]
    ),
    (
        ols_1997WithoutLagAll,
        "OLS 1997 Without Lagged FEDFUNDS (All Quarters)",
        ["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]
    ),
    (
        ols_1997WithLagQ1,
        "OLS 1997 With Lagged FEDFUNDS (Q1)",
        ["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]
    ),
    (
        ols_1997WithLagAll,
        "OLS 1997 With Lagged FEDFUNDS (All Quarters)",
        ["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]
    ),
    (
        ols_2002WithoutLagQ1,
        "OLS 2002 Without Lagged FEDFUNDS (Q1)",
        ["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]
    ),
    (
        ols_2002WithoutLagAll,
        "OLS 2002 Without Lagged FEDFUNDS (All Quarters)",
        ["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]
    ),
    (
        ols_2002WithLagQ1,
        "OLS 2002 With Lagged FEDFUNDS (Q1)",
        ["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]
    ),
    (
        ols_2002WithLagAll,
        "OLS 2002 With Lagged FEDFUNDS (All Quarters)",
        ["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]
    )
]

ivModels = [
    (
        results_1997WithoutLaggedQ1,
        "IV 1997 Without Lagged FEDFUNDS (Q1)",
        [],
        ["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"],
        ["Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
         "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
         "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3"]
    ),
    (
        results_1997WithoutLaggedAll,
        "IV 1997 Without Lagged FEDFUNDS (All Quarters)",
        [],
        ["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"],
        ["Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
         "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
         "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3"]
    ),
    (
        results_1997_withLaggedQ1,
        "IV 1997 With Lagged FEDFUNDS (Q1)",
        ["FedFunds_1997_Lag1"],
        ["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"],
        ["Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
         "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
         "Houses_1997_Lag1"]
    ),
    (
        results_1997_withLaggedAll,
        "IV 1997 With Lagged FEDFUNDS (All Quarters)",
        ["FedFunds_1997_Lag1"],
        ["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"],
        ["Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
         "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
         "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3"]
    ),
    (
        results_2002WithoutLaggedQ1,
        "IV 2002 Without Lagged FEDFUNDS (Q1)",
        [],
        ["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"],
        ["Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
         "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
         "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3"]
    ),
    (
        results_2002WithoutLaggedAll,
        "IV 2002 Without Lagged FEDFUNDS (All Quarters)",
        [],
        ["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"],
        ["Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
         "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
         "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3"]
    ),
    (
        results_2002_withLaggedQ1,
        "IV 2002 With Lagged FEDFUNDS (Q1)",
        ["FedFunds_2002_Lag1"],
        ["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"],
        ["Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
         "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
         "Houses_2002_Lag1"]
    ),
    (
        results_2002_withLaggedAll,
        "IV 2002 With Lagged FEDFUNDS (All Quarters)",
        ["FedFunds_2002_Lag1"],
        ["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"],
        ["Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
         "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
         "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3"]
    )
]

predictions = {}

def evaluateModels(testData, olsModels, ivModels):
    evaluationResults = []

    for model, modelName, features in olsModels:
        X_test = testData[features]
        X_test = add_constant(X_test)
        y_test = testData["FEDFUNDS_19970107"]
        print(len(y_test))

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - len(features) - 1))
        variance = y_test.var()
        mse_mean_ratio = mse / y_test.mean()

        evaluationResults.append({
            "Model": modelName,
            "Type": "OLS",
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "Adjusted R2": adj_r2,
            "Variance": variance,
            "MSE/Variance": mse / variance,
            "MSE/Mean": mse_mean_ratio
        })

        predictions[modelName] = y_pred

    for model, modelName, exog_features, endog_features, instruments in ivModels:
        print(f"Evaluating IV model: {modelName}")
        exog_test = add_constant(testData[exog_features])
        endog_test = testData[endog_features]
        y_test = testData["FEDFUNDS_19970107"]

        y_pred = model.predict(exog=exog_test, endog=endog_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - len(exog_features) - 1))
        variance = y_test.var()
        mse_mean_ratio = mse / y_test.mean()

        evaluationResults.append({
            "Model": modelName,
            "Type": "IV",
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "Adjusted R2": adj_r2,
            "Variance": variance,
            "MSE/Variance": mse / variance,
            "MSE/Mean": mse_mean_ratio
        })
        predictions[modelName] = y_pred

    return pd.DataFrame(evaluationResults)

modelComparisonResults = evaluateModels(mergedTestData, olsModels, ivModels)
print(modelComparisonResults)

dates = mergedTestData["observation_date"]
actual = mergedTestData["FEDFUNDS_19970107"]
predicted = predictions["OLS 1997 With Lagged FEDFUNDS (All Quarters)"]
residuals = actual - predicted

plt.figure(figsize=(12, 4))
plt.plot(dates, residuals, marker='o', linestyle='-', color="darkgreen")
plt.axhline(0, color='black', linestyle='--')
plt.title("Residuals Over Time – OLS 1997 With Lagged FEDFUNDS (All Quarters) [Only Houses]")
plt.xlabel("Date")
plt.ylabel("Prediction Error (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()