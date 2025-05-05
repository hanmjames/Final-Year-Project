import pandas as pd
import numpy as np
import joblib
from statsmodels.api import add_constant
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option('display.max_columns', None)

from Model_Training.RegressionAnalysis import ols_2002WithoutLagAll, ols_1997WithoutLagAll, ols_2002WithLagAll, ols_2002WithLagQ1, ols_1997WithLagAll, ols_1997WithLagQ1, ols_2002WithoutLagQ1, ols_1997WithoutLagQ1, results2002WithoutLaggedAll, results_1997WithLaggedAll, results_2002WithLaggedQ1, results_1997WithoutLaggedAll, results_2002WithLaggedAll, results2002WithoutLaggedQ1, results_1997WithoutLaggedQ1, results_1997WithLaggedQ1

fedFundsTest = pd.read_csv("../Training_and_Testing_Data/FedfundsTest.csv")
inflationTest = pd.read_csv("../Training_and_Testing_Data/InflationTest.csv")
realGDPTest = pd.read_csv("../Training_and_Testing_Data/RealGDPTest.csv")
potGDPTest = pd.read_csv("../Training_and_Testing_Data/PotGDPTest.csv")

fedFundsTest['observation_date'] = pd.to_datetime(fedFundsTest['observation_date'])
inflationTest['observation_date'] = pd.to_datetime(inflationTest['observation_date'])
realGDPTest['observation_date'] = pd.to_datetime(realGDPTest['observation_date'])
potGDPTest['observation_date'] = pd.to_datetime(potGDPTest['observation_date'])

startDate = '1998-01-01'
endDate = '2006-12-31'

fedFundsTest = fedFundsTest[(fedFundsTest['observation_date'] >= startDate) & (fedFundsTest['observation_date'] <= endDate)]
inflationTest = inflationTest[(inflationTest['observation_date'] >= startDate) & (inflationTest['observation_date'] <= endDate)]
realGDPTest = realGDPTest[(realGDPTest['observation_date'] >= startDate) & (realGDPTest['observation_date'] <= endDate)]
potGDPTest = potGDPTest[(potGDPTest['observation_date'] >= startDate) & (potGDPTest['observation_date'] <= endDate)]

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

mergedTestData = (
    fedFundsTest[['observation_date', 'FEDFUNDS', 'FEDFUNDS_Lag1', 'FEDFUNDS_Lag2', 'FEDFUNDS_Lag3']]
    .merge(inflationTest[['observation_date', 'Inflation_Rate', 'Inflation_Rate_Lag1', 'Inflation_Rate_Lag2', 'Inflation_Rate_Lag3']], on='observation_date')
    .merge(mergedPotGDP[['observation_date', 'OutputGap', 'OutputGap_Lag1', 'OutputGap_Lag2', 'OutputGap_Lag3']], on='observation_date')
)
columnRenameMap = {
    'FEDFUNDS': ['FEDFUNDS_19970107', 'FEDFUNDS_20020108'],
    'Inflation_Rate': ['Inflation_Rate_1997', 'Inflation_Rate_2002'],
    'OutputGap': ['OutputGap_1997', 'OutputGap_2002'],
    'FEDFUNDS_Lag1': ['FedFunds_1997_Lag1', 'FedFunds_2002_Lag1'],
    'OutputGap_Lag1': ['OutputGap_1997_Lag1', 'OutputGap_2002_Lag1'],
    'OutputGap_Lag2': ['OutputGap_1997_Lag2', 'OutputGap_2002_Lag2'],
    'OutputGap_Lag3': ['OutputGap_1997_Lag3', 'OutputGap_2002_Lag3'],
    'Inflation_Rate_Lag1': ['Inflation_Rate_1997_Lag1', 'Inflation_Rate_2002_Lag1'],
    'Inflation_Rate_Lag2': ['Inflation_Rate_1997_Lag2', 'Inflation_Rate_2002_Lag2'],
    'Inflation_Rate_Lag3': ['Inflation_Rate_1997_Lag3', 'Inflation_Rate_2002_Lag3'],
}

for old_name, new_names in columnRenameMap.items():
    for new_name in new_names:
        mergedTestData[new_name] = mergedTestData[old_name]

standardizeCols = joblib.load("../Model_Training/BaseModelScalerColumns.joblib")
scaler = joblib.load("../Model_Training/BaseModelScaler.joblib")
mergedTestData.dropna(subset=standardizeCols, inplace=True)
mergedTestData[standardizeCols] = scaler.transform(mergedTestData[standardizeCols])

olsModels = [
    (
        ols_1997WithoutLagQ1,
        "OLS 1997 Without Lagged FEDFUNDS (Q1)",
        ["Inflation_Rate", "OutputGap"]
    ),
    (
        ols_1997WithoutLagAll,
        "OLS 1997 Without Lagged FEDFUNDS (All Quarters)",
        ["Inflation_Rate", "OutputGap"]
    ),
    (
        ols_1997WithLagQ1,
        "OLS 1997 With Lagged FEDFUNDS (Q1)",
        ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"]
    ),
    (
        ols_1997WithLagAll,
        "OLS 1997 With Lagged FEDFUNDS (All Quarters)",
        ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"]
    ),
    (
        ols_2002WithoutLagQ1,
        "OLS 2002 Without Lagged FEDFUNDS (Q1)",
        ["Inflation_Rate", "OutputGap"]
    ),
    (
        ols_2002WithoutLagAll,
        "OLS 2002 Without Lagged FEDFUNDS (All Quarters)",
        ["Inflation_Rate", "OutputGap"]
    ),
    (
        ols_2002WithLagQ1,
        "OLS 2002 With Lagged FEDFUNDS (Q1)",
        ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"]
    ),
    (
        ols_2002WithLagAll,
        "OLS 2002 With Lagged FEDFUNDS (All Quarters)",
        ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"]
    )
]

ivModels = [
    (
        results_1997WithoutLaggedQ1,
        "IV 1997 Without Lagged FEDFUNDS (Q1)",
        [],
        ["Inflation_Rate", "OutputGap"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]
    ),
    (
        results_1997WithoutLaggedAll,
        "IV 1997 Without Lagged FEDFUNDS (All Quarters)",
        [],
        ["Inflation_Rate", "OutputGap"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]
    ),
    (
        results_1997WithLaggedQ1,
        "IV 1997 With Lagged FEDFUNDS (Q1)",
        ["FEDFUNDS_Lag1"],
        ["Inflation_Rate", "OutputGap"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]
    ),
    (
        results_1997WithLaggedAll,
        "IV 1997 With Lagged FEDFUNDS (All Quarters)",
        ["FEDFUNDS_Lag1"],
        ["Inflation_Rate", "OutputGap"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]
    ),
    (
        results2002WithoutLaggedQ1,
        "IV 2002 Without Lagged FEDFUNDS (Q1)",
        [],
        ["Inflation_Rate", "OutputGap"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]
    ),
    (
        results2002WithoutLaggedAll,
        "IV 2002 Without Lagged FEDFUNDS (All Quarters)",
        [],
        ["Inflation_Rate", "OutputGap"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]
    ),
    (
        results_2002WithLaggedQ1,
        "IV 2002 With Lagged FEDFUNDS (Q1)",
        ["FEDFUNDS_Lag1"],
        ["Inflation_Rate", "OutputGap"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]
    ),
    (
        results_2002WithLaggedAll,
        "IV 2002 With Lagged FEDFUNDS (All Quarters)",
        ["FEDFUNDS_Lag1"],
        ["Inflation_Rate", "OutputGap"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]
    )
]

predictions = {}

def evaluateModels(test_data, olsModels, ivModels):
    evaluationResults = []

    for model, modelName, features in olsModels:
        X_test = test_data[features]
        X_test = add_constant(X_test)  
        y_test = test_data["FEDFUNDS"]

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
        exog_test = add_constant(test_data[exog_features])  
        endog_test = test_data[endog_features]  
        y_test = test_data["FEDFUNDS"]

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

    return pd.DataFrame(evaluationResults), predictions

modelComparisonResults = evaluateModels(mergedTestData, olsModels, ivModels)
print(modelComparisonResults)

def computeVIF(df, features):
    X = add_constant(df[features])
    VIFDf = pd.DataFrame()
    VIFDf["Variable"] = X.columns
    VIFDf["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return VIFDf

VIFResult = computeVIF(mergedTestData, ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"])
print(VIFResult)

olsModelsAllQ = [m for m in olsModels if "All Quarters" in m[1]]
ivModelsAllQ = [m for m in ivModels if "All Quarters" in m[1]]

modelComparisonAllQ, predictionsAllQ = evaluateModels(mergedTestData, olsModelsAllQ, ivModelsAllQ)
predictionsAllQ = {k: v for k, v in predictionsAllQ.items() if "All Quarters" in k}
print(modelComparisonAllQ)

modelName = "IV 2002 With Lagged FEDFUNDS (All Quarters)"
actual = mergedTestData["FEDFUNDS"]
predicted = predictionsAllQ[modelName].squeeze()
dates = mergedTestData["observation_date"]

residuals = actual - predicted
plt.figure(figsize=(12, 4))
plt.plot(dates, residuals, marker='o', linestyle='-', color="red")
plt.axhline(0, color='black', linestyle='--')
plt.title("Residuals Over Time – IV 2002 With Lagged FEDFUNDS")
plt.xlabel("Date")
plt.ylabel("Prediction Error (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()