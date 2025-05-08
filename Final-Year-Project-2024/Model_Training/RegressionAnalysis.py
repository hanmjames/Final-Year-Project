import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS, add_constant
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import resample
pd.set_option('display.max_columns', None)

realGDP = pd.read_csv(r"../Training_and_Testing_Data/RealGDP_1981to1996.csv")
realGDP['observation_date'] = pd.to_datetime(realGDP['observation_date'], format='%d/%m/%Y')

potGDP = pd.read_csv(r"../Training_and_Testing_Data/PotGDP_1981to1996.csv")
potGDP['observation_date'] = pd.to_datetime(potGDP['observation_date'], format='%d/%m/%Y')

inflation = pd.read_csv(r"../Training_and_Testing_Data/Inflation_1981to1996.csv")
inflation['observation_date'] = pd.to_datetime(inflation['observation_date'], format='%d/%m/%Y')

fedFunds = pd.read_csv(r"../Training_and_Testing_Data/Fedfunds_1981to1996.csv")
fedFunds['observation_date'] = pd.to_datetime(fedFunds['observation_date'], format='%d/%m/%Y')

mergedData = (
    realGDP.merge(potGDP, on="observation_date")
    .merge(inflation, on="observation_date")
    .merge(fedFunds, on="observation_date")
)

mergedData['observation_date'] = pd.to_datetime(mergedData['observation_date'])
mergedData['Inflation_Rate_1997'] = (
    (mergedData['GDPDEF_19970131'] / mergedData['GDPDEF_19970131'].shift(4) - 1) * 100
)
mergedData['Inflation_Rate_2002'] = (
    (mergedData['GDPDEF_20020130'] / mergedData['GDPDEF_20020130'].shift(4) - 1) * 100
)
mergedData['OutputGap_1997'] = 100 * (
    np.log(mergedData['GDPC1_19970131']) - np.log(mergedData['GDPPOT_19970128'])
)
mergedData['OutputGap_2002'] = 100 * (
    np.log(mergedData['GDPC1_20020130']) - np.log(mergedData['GDPPOT_20020201'])
)

for lag in range(1, 4):
    mergedData[f'OutputGap_1997_Lag{lag}'] = mergedData['OutputGap_1997'].shift(lag)
    mergedData[f'Inflation_Rate_1997_Lag{lag}'] = mergedData['Inflation_Rate_1997'].shift(lag)
    mergedData[f'FedFunds_1997_Lag{lag}'] = mergedData['FEDFUNDS_19970107'].shift(lag)
    mergedData[f'OutputGap_2002_Lag{lag}'] = mergedData['OutputGap_2002'].shift(lag)
    mergedData[f'Inflation_Rate_2002_Lag{lag}'] = mergedData['Inflation_Rate_2002'].shift(lag)
    mergedData[f'FedFunds_2002_Lag{lag}'] = mergedData['FEDFUNDS_20020108'].shift(lag)

mergedData = mergedData[(mergedData['observation_date'] >= '1981-01-01') & (mergedData['observation_date'] <= '1996-12-31')]
mergedData.reset_index(drop=True, inplace=True)

print(mergedData.head())

standardizeCols = [
    "FEDFUNDS_19970107", "Inflation_Rate_1997", "OutputGap_1997",
    "FedFunds_1997_Lag1", "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "FEDFUNDS_20020108", "Inflation_Rate_2002", "OutputGap_2002",
    "FedFunds_2002_Lag1", "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3"
]

mergedData.dropna(subset=standardizeCols, inplace=True)
mergedDataQ1 = mergedData[mergedData['observation_date'].dt.strftime('%m-%d') == '01-01'].copy()

scaler = StandardScaler()
mergedData[standardizeCols] = scaler.fit_transform(mergedData[standardizeCols])
mergedDataQ1[standardizeCols] = scaler.transform(mergedDataQ1[standardizeCols])
mergedDataQ1 = mergedData.loc[mergedData['observation_date'].dt.strftime('%m-%d') == '01-01'].copy()
joblib.dump(scaler, 'BaseModelScaler.joblib')
joblib.dump(standardizeCols, 'BaseModelScalerColumns.joblib')

def saveSummary(summary, title, file_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0, 0, str(summary), fontsize=10, family="monospace")
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    plt.savefig(file_name, bbox_inches="tight")

y_1997Q1 = mergedDataQ1["FEDFUNDS_19970107"]
X_1997Q1 = mergedDataQ1[["Inflation_Rate_1997", "OutputGap_1997"]]
X_1997Q1 = add_constant(X_1997Q1)

ols_1997WithoutLagQ1 = OLS(y_1997Q1, X_1997Q1).fit()
print("OLS Results for 1997 (Without Lagged FedFunds, First Quarter Only):")
print(ols_1997WithoutLagQ1.summary())

y_1997All = mergedData["FEDFUNDS_19970107"]
X_1997All = mergedData[["Inflation_Rate_1997", "OutputGap_1997"]]
X_1997All = add_constant(X_1997All)

y_1997Q1 = mergedDataQ1["FEDFUNDS_19970107"]
X_1997Q1 = mergedDataQ1[["Inflation_Rate_1997", "OutputGap_1997"]]
X_1997Q1 = add_constant(X_1997Q1)

ols_1997WithoutLagQ1 = OLS(y_1997Q1, X_1997Q1).fit()
print("OLS Results for 1997 (Without Lagged FedFunds, First Quarter Only):")
print(ols_1997WithoutLagQ1.summary())

y_1997All = mergedData["FEDFUNDS_19970107"]
X_1997All = mergedData[["Inflation_Rate_1997", "OutputGap_1997"]]
X_1997All = add_constant(X_1997All)

ols_1997WithoutLagAll = OLS(y_1997All, X_1997All).fit()
print("OLS Results for 1997 (Without Lagged FedFunds, All Quarters):")
print(ols_1997WithoutLagAll.summary())

y_1997Q1Lagged = mergedDataQ1["FEDFUNDS_19970107"]
X_1997Q1Lagged = add_constant(mergedDataQ1[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]])

ols_1997WithLagQ1 = OLS(y_1997Q1Lagged, X_1997Q1Lagged).fit()
print("OLS Results for 1997 (With Lagged FedFunds, First Quarter Only):")
print(ols_1997WithLagQ1.summary())

y_1997AllLagged = mergedData["FEDFUNDS_19970107"]
X_1997AllLagged = add_constant(mergedData[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]])

ols_1997WithLagAll = OLS(y_1997AllLagged, X_1997AllLagged).fit()
print("OLS Results for 1997 (With Lagged FedFunds, All Quarters):")
print(ols_1997WithLagAll.summary())

y_2002Q1 = mergedDataQ1["FEDFUNDS_20020108"]
X_2002Q1 = mergedDataQ1[["Inflation_Rate_2002", "OutputGap_2002"]]
X_2002Q1 = add_constant(X_2002Q1)

ols_2002WithoutLagQ1 = OLS(y_2002Q1, X_2002Q1).fit()
print("OLS Results for 2002 (Without Lagged FedFunds, First Quarter Only):")
print(ols_2002WithoutLagQ1.summary())

y_2002All = mergedData["FEDFUNDS_20020108"]
X_2002All = mergedData[["Inflation_Rate_2002", "OutputGap_2002"]]
X_2002All = add_constant(X_2002All)

ols_2002WithoutLagAll = OLS(y_2002All, X_2002All).fit()
print("OLS Results for 2002 (Without Lagged FedFunds, All Quarters):")
print(ols_2002WithoutLagAll.summary())

y_2002Q1Lagged = mergedDataQ1["FEDFUNDS_20020108"]
X_2002Q1Lagged = add_constant(mergedDataQ1[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002"]])

ols_2002WithLagQ1 = OLS(y_2002Q1Lagged, X_2002Q1Lagged).fit()
print("OLS Results for 2002 (With Lagged FedFunds, First Quarter Only):")
print(ols_2002WithLagQ1.summary())

y_2002AllLagged = mergedData["FEDFUNDS_20020108"]
X_2002AllLagged = add_constant(mergedData[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002"]])

ols_2002WithLagAll = OLS(y_2002AllLagged, X_2002AllLagged).fit()
print("OLS Results for 2002 (With Lagged FedFunds, All Quarters):")
print(ols_2002WithLagAll.summary())

dependent_1997Q1 = mergedDataQ1["FEDFUNDS_19970107"]
exog_1997Q1 = add_constant(mergedDataQ1[[]])  
endog_1997Q1 = mergedDataQ1[["Inflation_Rate_1997", "OutputGap_1997"]]
instruments_1997Q1 = mergedDataQ1[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3"
]]
ivModel1997Q1 = IV2SLS(dependent_1997Q1, exog_1997Q1, endog_1997Q1, instruments_1997Q1)
results_1997WithoutLaggedQ1 = ivModel1997Q1.fit()
print("IV Regression Results for 1997 Vintage Excl. Lagged FEDFUNDS (First Quarter Only):")
print(results_1997WithoutLaggedQ1.summary)

dependent_1997All = mergedData["FEDFUNDS_19970107"]
exog_1997All = add_constant(mergedData[[]])  
endog_1997All = mergedData[["Inflation_Rate_1997", "OutputGap_1997"]]
instruments_1997All = mergedData[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3"
]]
ivModel1997All = IV2SLS(dependent_1997All, exog_1997All, endog_1997All, instruments_1997All)
results_1997WithoutLaggedAll = ivModel1997All.fit()
print("IV Regression Results for 1997 Vintage Excl. Lagged FEDFUNDS (All Quarters):")
print(results_1997WithoutLaggedAll.summary)

dependent_1997Q1Lagged = mergedDataQ1["FEDFUNDS_19970107"]
exog_1997Q1Lagged = add_constant(mergedDataQ1[["FedFunds_1997_Lag1"]])
endog_1997Q1Lagged = mergedDataQ1[["Inflation_Rate_1997", "OutputGap_1997"]]
instruments_1997Q1Lagged = mergedDataQ1[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3"
]]
ivModel1997Q1Lagged = IV2SLS(dependent_1997Q1Lagged, exog_1997Q1Lagged, endog_1997Q1Lagged, instruments_1997Q1Lagged)
results_1997WithLaggedQ1 = ivModel1997Q1Lagged.fit()
print("IV Regression Results for 1997 Vintage Incl. Lagged FEDFUNDS (First Quarter Only):")
print(results_1997WithLaggedQ1.summary)

dependent_1997AllLagged = mergedData["FEDFUNDS_19970107"]
exog_1997AllLagged = add_constant(mergedData[["FedFunds_1997_Lag1"]])
endog_1997AllLagged = mergedData[["Inflation_Rate_1997", "OutputGap_1997"]]
instruments_1997AllLagged = mergedData[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3"
]]
ivModel1997AllLagged = IV2SLS(dependent_1997AllLagged, exog_1997AllLagged, endog_1997AllLagged, instruments_1997AllLagged)
results_1997WithLaggedAll = ivModel1997AllLagged.fit()
print("IV Regression Results for 1997 Vintage Incl. Lagged FEDFUNDS (All Quarters):")
print(results_1997WithLaggedAll.summary)

dependent_2002Q1 = mergedDataQ1["FEDFUNDS_20020108"]
exog_2002Q1 = add_constant(mergedDataQ1[[]])  
endog_2002Q1 = mergedDataQ1[["Inflation_Rate_2002", "OutputGap_2002"]]
instruments_2002Q1 = mergedDataQ1[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3"
]]
ivModel2002Q1 = IV2SLS(dependent_2002Q1, exog_2002Q1, endog_2002Q1, instruments_2002Q1)
results2002WithoutLaggedQ1 = ivModel2002Q1.fit()
print("IV Regression Results for 2002 Vintage Excl. Lagged FEDFUNDS (First Quarter Only):")
print(results2002WithoutLaggedQ1.summary)

dependent_2002All = mergedData["FEDFUNDS_20020108"]
exog_2002All = add_constant(mergedData[[]])  
endog_2002All = mergedData[["Inflation_Rate_2002", "OutputGap_2002"]]
instruments_2002All = mergedData[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3"
]]
ivModel2002All = IV2SLS(dependent_2002All, exog_2002All, endog_2002All, instruments_2002All)
results2002WithoutLaggedAll = ivModel2002All.fit()
print("IV Regression Results for 2002 Vintage Excl. Lagged FEDFUNDS (All Quarters):")
print(results2002WithoutLaggedAll.summary)

dependent_2002Q1Lagged = mergedDataQ1["FEDFUNDS_20020108"]
exog_2002Q1Lagged = add_constant(mergedDataQ1[["FedFunds_2002_Lag1"]])
endog_2002Q1Lagged = mergedDataQ1[["Inflation_Rate_2002", "OutputGap_2002"]]
instruments_2002Q1Lagged = mergedDataQ1[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3"
]]
ivModel2002Q1Lagged = IV2SLS(dependent_2002Q1Lagged, exog_2002Q1Lagged, endog_2002Q1Lagged, instruments_2002Q1Lagged)
results_2002WithLaggedQ1 = ivModel2002Q1Lagged.fit()
print("IV Regression Results for 2002 Vintage Incl. Lagged FEDFUNDS (First Quarter Only):")
print(results_2002WithLaggedQ1.summary)

dependent_2002AllLagged = mergedData["FEDFUNDS_20020108"]
exog_2002AllLagged = add_constant(mergedData[["FedFunds_2002_Lag1"]])
endog_2002AllLagged = mergedData[["Inflation_Rate_2002", "OutputGap_2002"]]
instruments_2002AllLagged = mergedData[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3"
]]
ivModel2002AllLagged = IV2SLS(dependent_2002AllLagged, exog_2002AllLagged, endog_2002AllLagged, instruments_2002AllLagged)
results_2002WithLaggedAll = ivModel2002AllLagged.fit()
print("IV Regression Results for 2002 Vintage Incl. Lagged FEDFUNDS (All Quarters):")
print(results_2002WithLaggedAll.summary)

# windowSize = 20
# coeffList = []
# dateList = []
#
# for i in range(windowSize, len(mergedData)):
#     windowData = mergedData.iloc[i - windowSize:i]
#     y = windowData["FEDFUNDS_19970107"]
#     X = windowData[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]]
#     X = add_constant(X)
#     model = OLS(y, X).fit()
#     coeffs = model.params
#     coeffList.append(coeffs)
#     dateList.append(windowData["observation_date"].iloc[-1])
#
# coeffDf1 = pd.DataFrame(coeffList)
# coeffDf1["observation_date"] = dateList
# print(coeffDf1)
#
# plt.figure(figsize=(12, 6))
# for col in coeffDf1.columns:
#     if col != 'observation_date':
#         plt.plot(coeffDf1['observation_date'], coeffDf1[col], label=col)
#
# plt.title("Rolling OLS Coefficients (1997 Model with Lagged FedFunds)")
# plt.xlabel("Date")
# plt.ylabel("Coefficient Value")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

results = [
    {"Vintage": "1997", "Regression Type": "IV with Lagged FEDFUNDS (1997)", "Model": results_1997WithLaggedQ1},
    {"Vintage": "1997", "Regression Type": "IV without Lagged FEDFUNDS (1997)", "Model": results_1997WithoutLaggedQ1},
    {"Vintage": "1997", "Regression Type": "IV with Lagged FEDFUNDS (1997, All Quarters)", "Model": results_1997WithLaggedAll},
    {"Vintage": "1997", "Regression Type": "IV without Lagged FEDFUNDS (1997, All Quarters)", "Model": results_1997WithoutLaggedAll},
    {"Vintage": "2002", "Regression Type": "IV with Lagged FEDFUNDS (2002)", "Model": results_2002WithLaggedQ1},
    {"Vintage": "2002", "Regression Type": "IV without Lagged FEDFUNDS (2002)", "Model": results2002WithoutLaggedQ1},
    {"Vintage": "2002", "Regression Type": "IV with Lagged FEDFUNDS (2002, All Quarters)", "Model": results_2002WithLaggedAll},
    {"Vintage": "2002", "Regression Type": "IV without Lagged FEDFUNDS (2002, All Quarters)", "Model": results2002WithoutLaggedAll},
]

tableData = []
for res in results:
    model = res["Model"]
    tableData.append({
        "Vintage": res["Vintage"],
        "Regression Type": res["Regression Type"],
        "No. of Observations": model.nobs,
        "R-squared": model.rsquared,
        "F-statistic": model.f_statistic.stat,
        "P-value (F-stat)": model.f_statistic.pval,
        "Sargan Test Statistic": model.sargan.stat if model.sargan is not None else "N/A",
        "Sargan P-value": model.sargan.pval if model.sargan is not None else "N/A"
    })

ivComparisonDf = pd.DataFrame(tableData)
print(ivComparisonDf)

olsResultsData = []

models = [
    (ols_1997WithoutLagQ1, "1997", "OLS Without Lagged FEDFUNDS (First Quarter Only)"),
    (ols_1997WithoutLagAll, "1997", "OLS Without Lagged FEDFUNDS (All Quarters)"),
    (ols_1997WithLagQ1, "1997", "OLS With Lagged FEDFUNDS (First Quarter Only)"),
    (ols_1997WithLagAll, "1997", "OLS With Lagged FEDFUNDS (All Quarters)"),
    (ols_2002WithoutLagQ1, "2002", "OLS Without Lagged FEDFUNDS (First Quarter Only)"),
    (ols_2002WithoutLagAll, "2002", "OLS Without Lagged FEDFUNDS (All Quarters)"),
    (ols_2002WithLagQ1, "2002", "OLS With Lagged FEDFUNDS (First Quarter Only)"),
    (ols_2002WithLagAll, "2002", "OLS With Lagged FEDFUNDS (All Quarters)")
]

for model, vintage, regression_type in models:
    olsResultsData.append({
        "Vintage": vintage,
        "Regression Type": regression_type,
        "No. of Observations": model.nobs,
        "R-squared": model.rsquared,
        "Adjusted R-squared": model.rsquared_adj,
        "F-statistic": model.fvalue,
        "P-value (F-stat)": model.f_pvalue,
        "AIC": model.aic,
        "BIC": model.bic,
        "Durbin-Watson": sm.stats.durbin_watson(model.resid)  
    })

olsResultsDf = pd.DataFrame(olsResultsData)
print(olsResultsDf)

trainDataQ1, testDataQ1 = train_test_split(mergedDataQ1, test_size=0.2, random_state=42)
trainData, testData = train_test_split(mergedData, test_size=0.2, random_state=42)

standardizeCols  = [
    "Inflation_Rate_1997", "OutputGap_1997", "FedFunds_1997_Lag1",
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Inflation_Rate_2002", "OutputGap_2002", "FedFunds_2002_Lag1",
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3"
]


def runOLS(xTrain, yTrain, xTest, yTest):
    xTrain = add_constant(xTrain)
    xTest = add_constant(xTest)
    model = OLS(yTrain, xTrain).fit()
    yTrainPred = model.predict(xTrain)
    yTestPred = model.predict(xTest)
    metrics = {
        "R-squared": model.rsquared,
        "Adj. R-squared": model.rsquared_adj,
        "F-statistic": model.fvalue,
        "P-value (F-stat)": model.f_pvalue,
        "AIC": model.aic,
        "BIC": model.bic,
        "Durbin-Watson": sm.stats.durbin_watson(model.resid),
        "In-sample RMSE": np.sqrt(mean_squared_error(yTrain, yTrainPred)),
        "In-sample MAE": mean_absolute_error(yTrain, yTrainPred),
        "Out-of-sample RMSE": np.sqrt(mean_squared_error(yTest, yTestPred)),
        "Out-of-sample MAE": mean_absolute_error(yTest, yTestPred)
    }
    return pd.DataFrame([metrics])

def runIV (model, xTrainExog, xTrainEndog, yTrain, xTestExog, xTestEndog, yTest):
    xTrainExog = add_constant(xTrainExog) if "const" in model.params.index else xTrainExog
    xTestExog = add_constant(xTestExog) if "const" in model.params.index else xTestExog
    yTrainPred = model.predict(exog=xTrainExog, endog=xTrainEndog)
    yTestPred = model.predict(exog=xTestExog, endog=xTestEndog)
    metrics = {
        "R-squared": model.rsquared,
        "F-statistic": model.f_statistic.stat,
        "P-value (F-stat)": model.f_statistic.pval,
        "Sargan Stat": model.sargan.stat if model.sargan is not None else "N/A",
        "Sargan P-value": model.sargan.pval if model.sargan is not None else "N/A",
        "In-sample RMSE": np.sqrt(mean_squared_error(yTrain, yTrainPred)),
        "In-sample MAE": mean_absolute_error(yTrain, yTrainPred),
        "Out-of-sample RMSE": np.sqrt(mean_squared_error(yTest, yTestPred)),
        "Out-of-sample MAE": mean_absolute_error(yTest, yTestPred)
    }
    return pd.DataFrame([metrics])

allMetrics64 = []

allMetrics64.append(runOLS(
    trainData[["Inflation_Rate_1997", "OutputGap_1997"]],
    trainData["FEDFUNDS_19970107"],
    testData [["Inflation_Rate_1997", "OutputGap_1997"]],
    testData ["FEDFUNDS_19970107"]
).assign(Model="OLS 1997 Without Lagged FedFunds (All Quarters)"))

allMetrics64.append(runOLS(
    trainData[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]],
    trainData["FEDFUNDS_19970107"],
    testData [["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]],
    testData ["FEDFUNDS_19970107"]
).assign(Model="OLS 1997 With Lagged FedFunds (All Quarters)"))

allMetrics64.append(runOLS(
    trainData[["Inflation_Rate_2002", "OutputGap_2002"]],
    trainData["FEDFUNDS_20020108"],
    testData [["Inflation_Rate_2002", "OutputGap_2002"]],
    testData ["FEDFUNDS_20020108"]
).assign(Model="OLS 2002 Without Lagged FedFunds (All Quarters)"))

allMetrics64.append(runOLS(
    trainData[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002"]],
    trainData["FEDFUNDS_20020108"],
    testData [["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002"]],
    testData ["FEDFUNDS_20020108"]
).assign(Model="OLS 2002 With Lagged FedFunds (All Quarters)"))

allMetrics64.append(runIV (
    results_1997WithoutLaggedAll,
    trainData[[]],
    trainData[["Inflation_Rate_1997", "OutputGap_1997"]],
    trainData["FEDFUNDS_19970107"],
    testData [[]],
    testData [["Inflation_Rate_1997", "OutputGap_1997"]],
    testData ["FEDFUNDS_19970107"]
).assign(Model="IV 1997 Without Lagged FedFunds (All Quarters)"))

allMetrics64.append(runIV (
    results_1997WithLaggedAll,
    trainData[["FedFunds_1997_Lag1"]],
    trainData[["Inflation_Rate_1997", "OutputGap_1997"]],
    trainData["FEDFUNDS_19970107"],
    testData [["FedFunds_1997_Lag1"]],
    testData [["Inflation_Rate_1997", "OutputGap_1997"]],
    testData ["FEDFUNDS_19970107"]
).assign(Model="IV 1997 With Lagged FedFunds (All Quarters)"))

allMetrics64.append(runIV (
    results2002WithoutLaggedAll,
    trainData[[]],
    trainData[["Inflation_Rate_2002", "OutputGap_2002"]],
    trainData["FEDFUNDS_20020108"],
    testData [[]],
    testData [["Inflation_Rate_2002", "OutputGap_2002"]],
    testData ["FEDFUNDS_20020108"]
).assign(Model="IV 2002 Without Lagged FedFunds (All Quarters)"))

allMetrics64.append(runIV (
    results_2002WithLaggedAll,
    trainData[["FedFunds_2002_Lag1"]],
    trainData[["Inflation_Rate_2002", "OutputGap_2002"]],
    trainData["FEDFUNDS_20020108"],
    testData [["FedFunds_2002_Lag1"]],
    testData [["Inflation_Rate_2002", "OutputGap_2002"]],
    testData ["FEDFUNDS_20020108"]
).assign(Model="IV 2002 With Lagged FedFunds (All Quarters)"))

metrics64df = pd.concat(allMetrics64, ignore_index=True)
print("All Metrics OLS and IV")
print(metrics64df)

olsModels = [
    (ols_1997WithoutLagQ1, "1997", "OLS", True, False, "OLS 1997 Without Lagged FedFunds (Q1)"),
    (ols_1997WithLagQ1,    "1997", "OLS", True, True,  "OLS 1997 With Lagged FedFunds (Q1)"),
    (ols_1997WithoutLagAll,"1997", "OLS", False, False, "OLS 1997 Without Lagged FedFunds (All Quarters)"),
    (ols_1997WithLagAll,   "1997", "OLS", False, True,  "OLS 1997 With Lagged FedFunds (All Quarters)"),
    (ols_2002WithoutLagQ1, "2002", "OLS", True, False,  "OLS 2002 Without Lagged FedFunds (Q1)"),
    (ols_2002WithLagQ1,    "2002", "OLS", True, True,   "OLS 2002 With Lagged FedFunds (Q1)"),
    (ols_2002WithoutLagAll,"2002", "OLS", False, False, "OLS 2002 Without Lagged FedFunds (All Quarters)"),
    (ols_2002WithLagAll,   "2002", "OLS", False, True,  "OLS 2002 With Lagged FedFunds (All Quarters)")
]

ivModels = [
    (results_1997WithoutLaggedQ1,  "1997", "IV", True,  False, None, None, "IV 1997 Without Lagged FedFunds (Q1)"),
    (results_1997WithLaggedQ1,     "1997", "IV", True,  True,  None, None, "IV 1997 With Lagged FedFunds (Q1)"),
    (results_1997WithoutLaggedAll, "1997", "IV", False, False, None, None, "IV 1997 Without Lagged FedFunds (All Quarters)"),
    (results_1997WithLaggedAll,    "1997", "IV", False, True,  None, None, "IV 1997 With Lagged FedFunds (All Quarters)"),
    (results2002WithoutLaggedQ1,  "2002", "IV", True,  False, None, None, "IV 2002 Without Lagged FedFunds (Q1)"),
    (results_2002WithLaggedQ1,     "2002", "IV", True,  True,  None, None, "IV 2002 With Lagged FedFunds (Q1)"),
    (results2002WithoutLaggedAll, "2002", "IV", False, False, None, None, "IV 2002 Without Lagged FedFunds (All Quarters)"),
    (results_2002WithLaggedAll,    "2002", "IV", False, True,  None, None, "IV 2002 With Lagged FedFunds (All Quarters)")
]

nIterations = 1000
resultsList = []


def runOLSBootstrapOOB(fullData, yCol, xCols, label, random_state):
    sample = resample(fullData, replace=True, n_samples=len(fullData), random_state=random_state)
    oob_idx = fullData.index.difference(sample.index)
    oob_data = fullData.loc[oob_idx]
    if oob_data.empty:
        return {"Model": label, "OOB RMSE": np.nan, "OOB MAE": np.nan}
    X_train = add_constant(sample[xCols])
    y_train = sample[yCol]
    model = OLS(y_train, X_train).fit()
    X_oob = add_constant(oob_data[xCols])
    y_oob = oob_data[yCol]
    y_pred_oob = model.predict(X_oob)
    return {
        "Model": label,
        "OOB RMSE": np.sqrt(mean_squared_error(y_oob, y_pred_oob)),
        "OOB MAE": mean_absolute_error(y_oob, y_pred_oob)
    }


def runIVBootstrapOOB(fullData, yCol, exogCols, endogCols, instrCols, label, random_state):
    sample = resample(fullData, replace=True, n_samples=len(fullData), random_state=random_state)
    oob_idx = fullData.index.difference(sample.index)
    oob_data = fullData.loc[oob_idx]
    if oob_data.empty:
        return {"Model": label, "OOB RMSE": np.nan, "OOB MAE": np.nan}
    y_train = sample[yCol]
    exog_train = add_constant(sample[exogCols]) if exogCols else add_constant(pd.DataFrame(index=sample.index))
    endog_train = sample[endogCols]
    instr_train = sample[instrCols]
    model = IV2SLS(y_train, exog_train, endog_train, instr_train).fit()
    y_oob = oob_data[yCol]
    exog_oob = add_constant(oob_data[exogCols]) if exogCols else add_constant(pd.DataFrame(index=oob_data.index))
    endog_oob = oob_data[endogCols]
    try:
        y_pred_oob = model.predict(exog=exog_oob, endog=endog_oob)
        return {
            "Model": label,
            "OOB RMSE": np.sqrt(mean_squared_error(y_oob, y_pred_oob)),
            "OOB MAE": mean_absolute_error(y_oob, y_pred_oob)
        }
    except:
        return {"Model": label, "OOB RMSE": np.nan, "OOB MAE": np.nan}


for i in range(nIterations):
    resultsList.append(runOLSBootstrapOOB(mergedData, "FEDFUNDS_19970107",
                                          ["Inflation_Rate_1997", "OutputGap_1997"], "OLS 1997 Without Lag", i))

    resultsList.append(runOLSBootstrapOOB(mergedData, "FEDFUNDS_19970107",
                                          ["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"],
                                          "OLS 1997 With Lag", i))

    resultsList.append(runOLSBootstrapOOB(mergedData, "FEDFUNDS_20020108",
                                          ["Inflation_Rate_2002", "OutputGap_2002"], "OLS 2002 Without Lag", i))

    resultsList.append(runOLSBootstrapOOB(mergedData, "FEDFUNDS_20020108",
                                          ["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002"],
                                          "OLS 2002 With Lag", i))

    resultsList.append(runIVBootstrapOOB(mergedData, "FEDFUNDS_19970107", [],
                                         ["Inflation_Rate_1997", "OutputGap_1997"],
                                         ["OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
                                          "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2",
                                          "Inflation_Rate_1997_Lag3"],
                                         "IV 1997 Without Lag", i))

    resultsList.append(runIVBootstrapOOB(mergedData, "FEDFUNDS_19970107",
                                         ["FedFunds_1997_Lag1"],
                                         ["Inflation_Rate_1997", "OutputGap_1997"],
                                         ["OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
                                          "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2",
                                          "Inflation_Rate_1997_Lag3"],
                                         "IV 1997 With Lag", i))

    resultsList.append(runIVBootstrapOOB(mergedData, "FEDFUNDS_20020108", [],
                                         ["Inflation_Rate_2002", "OutputGap_2002"],
                                         ["OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
                                          "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2",
                                          "Inflation_Rate_2002_Lag3"],
                                         "IV 2002 Without Lag", i))

    resultsList.append(runIVBootstrapOOB(mergedData, "FEDFUNDS_20020108",
                                         ["FedFunds_2002_Lag1"],
                                         ["Inflation_Rate_2002", "OutputGap_2002"],
                                         ["OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
                                          "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2",
                                          "Inflation_Rate_2002_Lag3"],
                                         "IV 2002 With Lag", i))

bootstrapDf = pd.DataFrame(resultsList)
bootstrapSummary = bootstrapDf.groupby("Model").agg(["mean", "std"])
print(bootstrapSummary)
