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

oilPrices = pd.read_csv(r"../Training_and_Testing_Data/OilPrice_Quarterly.csv")
oilPrices['observation_date'] = pd.to_datetime(oilPrices['observation_date'], format='%d/%m/%Y')

unemploymentRates = pd.read_csv(r"../Training_and_Testing_Data/UnemploymentRate_Quarterly.csv")
unemploymentRates['observation_date'] = pd.to_datetime(unemploymentRates['observation_date'], format='%d/%m/%Y')

housing = pd.read_csv(r"../Training_and_Testing_Data/Housing_Quarterly.csv")
housing['observation_date'] = pd.to_datetime(housing['observation_date'], format='%d/%m/%Y')

oilPrices = oilPrices[pd.to_datetime(oilPrices['observation_date']) >= '1970-01-01']
unemploymentRates = unemploymentRates[pd.to_datetime(unemploymentRates['observation_date']) >= '1970-01-01']
housing = housing[pd.to_datetime(housing['observation_date']) >= '1970-01-01']

mergedData = (
    realGDP.merge(potGDP, on="observation_date")
    .merge(inflation, on="observation_date")
    .merge(fedFunds, on="observation_date")
    .merge(oilPrices, on="observation_date")
    .merge(housing, on="observation_date")
    .merge(unemploymentRates, on="observation_date")
)

mergedData['observation_date'] = pd.to_datetime(mergedData['observation_date'], dayfirst= True, errors='coerce')
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
    mergedData[f'Oil_Price_1997_Lag{lag}'] = mergedData['OILPRICE_19970303'].shift(lag)
    mergedData[f'Houses_1997_Lag{lag}'] = mergedData['HOUST_19970122'].shift(lag)
    mergedData[f'Unemployment_1997_Lag{lag}'] = mergedData['UNRATE_19970110'].shift(lag)
    mergedData[f'OutputGap_2002_Lag{lag}'] = mergedData['OutputGap_2002'].shift(lag)
    mergedData[f'Inflation_Rate_2002_Lag{lag}'] = mergedData['Inflation_Rate_2002'].shift(lag)
    mergedData[f'FedFunds_2002_Lag{lag}'] = mergedData['FEDFUNDS_20020108'].shift(lag)
    mergedData[f'Oil_Price_2002_Lag{lag}'] = mergedData['OILPRICE_20020102'].shift(lag)
    mergedData[f'Houses_2002_Lag{lag}'] = mergedData['HOUST_20020117'].shift(lag)
    mergedData[f'Unemployment_2002_Lag{lag}'] = mergedData['UNRATE_20020104'].shift(lag)

mergedData = mergedData[(mergedData['observation_date'] >= '1981-01-01') & (mergedData['observation_date'] <= '1996-12-31')]
mergedData.reset_index(drop=True, inplace=True)

print(mergedData.head())
standardizeCols = [
    "FEDFUNDS_19970107", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122",
    "FedFunds_1997_Lag1",
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3",
    "FEDFUNDS_20020108", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117",
    "FedFunds_2002_Lag1",
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
    "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3"
]

mergedData.dropna(subset=standardizeCols, inplace=True)
scaler = StandardScaler()
mergedData[standardizeCols] = scaler.fit_transform(mergedData[standardizeCols])
mergedDataQ1 = mergedData[mergedData['observation_date'].dt.strftime('%m-%d') == '01-01'].copy()
joblib.dump(scaler, '3AddnlVarsScaler.joblib')
joblib.dump(standardizeCols, '3AddnlVarsScalerColumns.joblib')

def saveSummary(summary, title, file_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0, 0, str(summary), fontsize=10, family="monospace")
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    plt.savefig(file_name, bbox_inches="tight")

y_1997Q1 = mergedDataQ1["FEDFUNDS_19970107"]
X_1997Q1 = mergedDataQ1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]]
X_1997Q1 = add_constant(X_1997Q1 )

ols_1997WithoutLagQ1 = OLS(y_1997Q1 , X_1997Q1 ).fit()
print("OLS Results for 1997 (Without Lagged FedFunds, First Quarter Only):")
print(ols_1997WithoutLagQ1.summary())

y_1997All = mergedData["FEDFUNDS_19970107"]
X_1997All = mergedData[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]]
X_1997All = add_constant(X_1997All)

ols_1997WithoutLagAll = OLS(y_1997All, X_1997All).fit()
print("OLS Results for 1997 (Without Lagged FedFunds, All Quarters):")
print(ols_1997WithoutLagAll.summary())

y_1997Q1Lagged = mergedDataQ1["FEDFUNDS_19970107"]
X_1997Q1Lagged = add_constant(mergedDataQ1[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]])

ols_1997WithLagQ1 = OLS(y_1997Q1Lagged, X_1997Q1Lagged).fit()
print("OLS Results for 1997 (With Lagged FedFunds, First Quarter Only):")
print(ols_1997WithLagQ1.summary())

y_1997AllLagged = mergedData["FEDFUNDS_19970107"]
X_1997AllLagged = add_constant(mergedData[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]])

ols_1997WithLagAll = OLS(y_1997AllLagged, X_1997AllLagged).fit()
print("OLS Results for 1997 (With Lagged FedFunds, All Quarters):")
print(ols_1997WithLagAll.summary())

y_2002Q1 = mergedDataQ1["FEDFUNDS_20020108"]
X_2002Q1 = mergedDataQ1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]]
X_2002Q1 = add_constant(X_2002Q1)

ols_2002WithoutLagQ1 = OLS(y_2002Q1, X_2002Q1).fit()
print("OLS Results for 2002 (Without Lagged FedFunds, First Quarter Only):")
print(ols_2002WithoutLagQ1.summary())

y_2002All = mergedData["FEDFUNDS_20020108"]
X_2002All = mergedData[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]]
X_2002All = add_constant(X_2002All)

ols_2002WithoutLagAll = OLS(y_2002All, X_2002All).fit()
print("OLS Results for 2002 (Without Lagged FedFunds, All Quarters):")
print(ols_2002WithoutLagAll.summary())

y_2002Q1Lagged = mergedDataQ1["FEDFUNDS_20020108"]
X_2002Q1Lagged = add_constant(mergedDataQ1[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]])

ols_2002WithLagQ1 = OLS(y_2002Q1Lagged, X_2002Q1Lagged).fit()
print("OLS Results for 2002 (With Lagged FedFunds, First Quarter Only):")
print(ols_2002WithLagQ1.summary())

y_2002AllLagged = mergedData["FEDFUNDS_20020108"]
X_2002AllLagged = add_constant(mergedData[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]])

ols_2002WithLagAll = OLS(y_2002AllLagged, X_2002AllLagged).fit()
print("OLS Results for 2002 (With Lagged FedFunds, All Quarters):")
print(ols_2002WithLagAll.summary())

dependent_1997Q1 = mergedDataQ1["FEDFUNDS_19970107"]
exog_1997Q1 = add_constant(mergedDataQ1[[]])
endog_1997Q1 = mergedDataQ1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]]
instruments_1997Q1 = mergedDataQ1[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3",
    "Oil_Price_1997_Lag1", "Oil_Price_1997_Lag2", "Oil_Price_1997_Lag3",
    "Unemployment_1997_Lag1", "Unemployment_1997_Lag2", "Unemployment_1997_Lag3"
]]
iv_model_1997Q1 = IV2SLS(dependent_1997Q1 , exog_1997Q1 , endog_1997Q1 , instruments_1997Q1 )
results1997WithoutLaggedQ1 = iv_model_1997Q1 .fit()
print("IV Regression Results for 1997 Vintage Excl. Lagged FEDFUNDS (First Quarter Only):")
print(results1997WithoutLaggedQ1.summary)

dependent_1997All = mergedData["FEDFUNDS_19970107"]
exog_1997All = add_constant(mergedData[[]])
endog_1997All = mergedData[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]]
instruments_1997All = mergedData[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3",
    "Oil_Price_1997_Lag1", "Oil_Price_1997_Lag2", "Oil_Price_1997_Lag3",
    "Unemployment_1997_Lag1", "Unemployment_1997_Lag2", "Unemployment_1997_Lag3"
]]
iv_model_1997All = IV2SLS(dependent_1997All, exog_1997All, endog_1997All, instruments_1997All)
results1997WithoutLaggedAll = iv_model_1997All.fit()
print("IV Regression Results for 1997 Vintage Excl. Lagged FEDFUNDS (All Quarters):")
print(results1997WithoutLaggedAll.summary)

dependent_1997Q1Lagged = mergedDataQ1["FEDFUNDS_19970107"]
exog_1997Q1Lagged = add_constant(mergedDataQ1[["FedFunds_1997_Lag1"]])
endog_1997Q1Lagged = mergedDataQ1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]]

instruments_1997Q1Lagged = mergedDataQ1[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Houses_1997_Lag1",
    "Oil_Price_1997_Lag1",
    "Unemployment_1997_Lag1"
]]
iv_model_1997Q1Lagged = IV2SLS(dependent_1997Q1Lagged, exog_1997Q1Lagged, endog_1997Q1Lagged, instruments_1997Q1Lagged)
results1997WithLaggedQ1 = iv_model_1997Q1Lagged.fit()
print("IV Regression Results for 1997 Vintage Incl. Lagged FEDFUNDS (First Quarter Only):")
print(results1997WithLaggedQ1.summary)

dependent_1997AllLagged = mergedData["FEDFUNDS_19970107"]
exog_1997AllLagged = add_constant(mergedData[["FedFunds_1997_Lag1"]])
endog_1997AllLagged = mergedData[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]]
instruments_1997AllLagged = mergedData[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3",
    "Oil_Price_1997_Lag1", "Oil_Price_1997_Lag2", "Oil_Price_1997_Lag3",
    "Unemployment_1997_Lag1", "Unemployment_1997_Lag2", "Unemployment_1997_Lag3"
]]
iv_model_1997AllLagged = IV2SLS(dependent_1997AllLagged, exog_1997AllLagged, endog_1997AllLagged, instruments_1997AllLagged)
results1997WithLaggedAll = iv_model_1997AllLagged.fit()
print("IV Regression Results for 1997 Vintage Incl. Lagged FEDFUNDS (All Quarters):")
print(results1997WithLaggedAll.summary)

dependent_2002Q1 = mergedDataQ1["FEDFUNDS_20020108"]
exog_2002Q1 = add_constant(mergedDataQ1[[]])
endog_2002Q1 = mergedDataQ1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]]
instruments_2002Q1 = mergedDataQ1[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
    "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3",
    "Oil_Price_2002_Lag1", "Oil_Price_2002_Lag2", "Oil_Price_2002_Lag3",
    "Unemployment_2002_Lag1", "Unemployment_2002_Lag2", "Unemployment_2002_Lag3"
]]
iv_model_2002Q1 = IV2SLS(dependent_2002Q1, exog_2002Q1, endog_2002Q1, instruments_2002Q1)
results2002WithoutLaggedQ1 = iv_model_2002Q1.fit()
print("IV Regression Results for 2002 Vintage Excl. Lagged FEDFUNDS (First Quarter Only):")
print(results2002WithoutLaggedQ1.summary)

dependent_2002All = mergedData["FEDFUNDS_20020108"]
exog_2002All = add_constant(mergedData[[]])
endog_2002All = mergedData[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]]
instruments_2002All = mergedData[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
    "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3",
    "Oil_Price_2002_Lag1", "Oil_Price_2002_Lag2", "Oil_Price_2002_Lag3",
    "Unemployment_2002_Lag1", "Unemployment_2002_Lag2", "Unemployment_2002_Lag3"
]]
iv_model_2002All = IV2SLS(dependent_2002All, exog_2002All, endog_2002All, instruments_2002All)
results2002WithoutLaggedAll = iv_model_2002All.fit()
print("IV Regression Results for 2002 Vintage Excl. Lagged FEDFUNDS (All Quarters):")
print(results2002WithoutLaggedAll.summary)

dependent_2002Q1Lagged = mergedDataQ1["FEDFUNDS_20020108"]
exog_2002Q1Lagged = add_constant(mergedDataQ1[["FedFunds_2002_Lag1"]])
endog_2002Q1Lagged = mergedDataQ1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]]

instruments_2002Q1Lagged = mergedDataQ1[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
    "Houses_2002_Lag1",
    "Oil_Price_2002_Lag1",
    "Unemployment_2002_Lag1"
]]
iv_model_2002Q1Lagged = IV2SLS(dependent_2002Q1Lagged, exog_2002Q1Lagged, endog_2002Q1Lagged, instruments_2002Q1Lagged)
results2002WithLaggedQ1 = iv_model_2002Q1Lagged.fit()
print("IV Regression Results for 2002 Vintage Incl. Lagged FEDFUNDS (First Quarter Only):")
print(results2002WithLaggedQ1.summary)

dependent_2002AllLagged = mergedData["FEDFUNDS_20020108"]
exog_2002AllLagged = add_constant(mergedData[["FedFunds_2002_Lag1"]])
endog_2002AllLagged = mergedData[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]]
instruments_2002AllLagged = mergedData[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
    "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3",
    "Oil_Price_2002_Lag1", "Oil_Price_2002_Lag2", "Oil_Price_2002_Lag3",
    "Unemployment_2002_Lag1", "Unemployment_2002_Lag2", "Unemployment_2002_Lag3"
]]
iv_model_2002AllLagged = IV2SLS(dependent_2002AllLagged, exog_2002AllLagged, endog_2002AllLagged, instruments_2002AllLagged)
results2002WithLaggedAll = iv_model_2002AllLagged.fit()
print("IV Regression Results for 2002 Vintage Incl. Lagged FEDFUNDS (All Quarters):")
print(results2002WithLaggedAll.summary)

results = [
    {"Vintage": "1997", "Regression Type": "IV with Lagged FEDFUNDS (1997)", "Model": results1997WithLaggedQ1},
    {"Vintage": "1997", "Regression Type": "IV without Lagged FEDFUNDS (1997)", "Model": results1997WithoutLaggedQ1},
    {"Vintage": "1997", "Regression Type": "IV with Lagged FEDFUNDS (1997, All Quarters)", "Model": results1997WithLaggedAll},
    {"Vintage": "1997", "Regression Type": "IV without Lagged FEDFUNDS (1997, All Quarters)", "Model": results1997WithoutLaggedAll},
    {"Vintage": "2002", "Regression Type": "IV with Lagged FEDFUNDS (2002)", "Model": results2002WithLaggedQ1},
    {"Vintage": "2002", "Regression Type": "IV without Lagged FEDFUNDS (2002)", "Model": results2002WithoutLaggedQ1},
    {"Vintage": "2002", "Regression Type": "IV with Lagged FEDFUNDS (2002, All Quarters)", "Model": results2002WithLaggedAll},
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

def calculateMetricsIV(model, X_trainExog, X_train_endog, y_train, X_testExog, X_test_endog, y_test, modelName):
    X_trainExog = add_constant(X_trainExog) if "const" in model.params.index else X_trainExog
    X_testExog = add_constant(X_testExog) if "const" in model.params.index else X_testExog
    y_trainPred = model.predict(exog=X_trainExog, endog=X_train_endog)
    y_testPred = model.predict(exog=X_testExog, endog=X_test_endog)

    metrics = {
        "Model": modelName,
        "In-sample R-squared": model.rsquared,
        "In-sample RMSE": np.sqrt(mean_squared_error(y_train, y_trainPred)),
        "In-sample MAE": mean_absolute_error(y_train, y_trainPred),
        "Out-of-sample RMSE": np.sqrt(mean_squared_error(y_test, y_testPred)),
        "Out-of-sample MAE": mean_absolute_error(y_test, y_testPred)
    }
    return metrics

def calculateMetrics(model, X_train, y_train, X_test, y_test, modelName):
    X_train = add_constant(X_train) if "const" in model.params.index else X_train
    X_test = add_constant(X_test) if "const" in model.params.index else X_test
    y_trainPred = model.predict(X_train)
    y_testPred = model.predict(X_test)

    metrics = {
        "Model": modelName,
        "In-sample R-squared": model.rsquared,
        "In-sample RMSE": np.sqrt(mean_squared_error(y_train, y_trainPred)),
        "In-sample MAE": mean_absolute_error(y_train, y_trainPred),
        "Out-of-sample RMSE": np.sqrt(mean_squared_error(y_test, y_testPred)),
        "Out-of-sample MAE": mean_absolute_error(y_test, y_testPred)
    }
    return metrics

olsModels = [
    (ols_1997WithoutLagQ1,
     trainDataQ1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],
     trainDataQ1["FEDFUNDS_19970107"],
     testDataQ1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],
     testDataQ1["FEDFUNDS_19970107"],
     "OLS 1997 Without Lagged FEDFUNDS (Q1)"),
    (ols_1997WithoutLagAll,
     trainData[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],
     trainData["FEDFUNDS_19970107"],
     testData[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],
     testData["FEDFUNDS_19970107"],
     "OLS 1997 Without Lagged FEDFUNDS (All Quarters)"),
    (ols_1997WithLagQ1,
     trainDataQ1[
         ["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303",
          "UNRATE_19970110"]],
     trainDataQ1["FEDFUNDS_19970107"],
     testDataQ1[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303",
                   "UNRATE_19970110"]],
     testDataQ1["FEDFUNDS_19970107"],
     "OLS 1997 With Lagged FEDFUNDS (Q1)"),
    (ols_1997WithLagAll,
     trainData[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303",
                 "UNRATE_19970110"]],
     trainData["FEDFUNDS_19970107"],
     testData[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303",
                "UNRATE_19970110"]],
     testData["FEDFUNDS_19970107"],
     "OLS 1997 With Lagged FEDFUNDS (All Quarters)"),
    (ols_2002WithoutLagQ1,
     trainDataQ1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],
     trainDataQ1["FEDFUNDS_20020108"],
     testDataQ1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],
     testDataQ1["FEDFUNDS_20020108"],
     "OLS 2002 Without Lagged FEDFUNDS (Q1)"),
    (ols_2002WithoutLagAll,
     trainData[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],
     trainData["FEDFUNDS_20020108"],
     testData[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],
     testData["FEDFUNDS_20020108"],
     "OLS 2002 Without Lagged FEDFUNDS (All Quarters)"),
    (ols_2002WithLagQ1,
     trainDataQ1[
         ["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102",
          "UNRATE_20020104"]],
     trainDataQ1["FEDFUNDS_20020108"],
     testDataQ1[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102",
                   "UNRATE_20020104"]],
     testDataQ1["FEDFUNDS_20020108"],
     "OLS 2002 With Lagged FEDFUNDS (Q1)"),
    (ols_2002WithLagAll,
     trainData[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102",
                 "UNRATE_20020104"]],
     trainData["FEDFUNDS_20020108"],
     testData[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102",
                "UNRATE_20020104"]],
     testData["FEDFUNDS_20020108"],
     "OLS 2002 With Lagged FEDFUNDS (All Quarters)")
]

ivModels = [
    (results1997WithoutLaggedQ1,
     trainDataQ1[[]],
     trainDataQ1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],
     trainDataQ1["FEDFUNDS_19970107"],
     testDataQ1[[]],
     testDataQ1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],
     testDataQ1["FEDFUNDS_19970107"],
     "IV 1997 Without Lagged FEDFUNDS (Q1)"
     ),

    (results1997WithoutLaggedAll,
     trainData[[]],
     trainData[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],

     trainData["FEDFUNDS_19970107"],
     testData[[]],
     testData[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],

     testData["FEDFUNDS_19970107"],
     "IV 1997 Without Lagged FEDFUNDS (All Quarters)"
     ),

    (results1997WithLaggedQ1,
     trainDataQ1[["FedFunds_1997_Lag1"]],
     trainDataQ1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],

     trainDataQ1["FEDFUNDS_19970107"],
     testDataQ1[["FedFunds_1997_Lag1"]],
     testDataQ1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],

     testDataQ1["FEDFUNDS_19970107"],
     "IV 1997 With Lagged FEDFUNDS (Q1)"
     ),

    (results1997WithLaggedAll,
     trainData[["FedFunds_1997_Lag1"]],
     trainData[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],

     trainData["FEDFUNDS_19970107"],
     testData[["FedFunds_1997_Lag1"]],
     testData[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"]],

     testData["FEDFUNDS_19970107"],
     "IV 1997 With Lagged FEDFUNDS (All Quarters)"
     ),

    (results2002WithoutLaggedQ1,
     trainDataQ1[[]],
     trainDataQ1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],

     trainDataQ1["FEDFUNDS_20020108"],
     testDataQ1[[]],
     testDataQ1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],

     testDataQ1["FEDFUNDS_20020108"],
     "IV 2002 Without Lagged FEDFUNDS (Q1)"
     ),

    (results2002WithoutLaggedAll,
     trainData[[]],
     trainData[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],

     trainData["FEDFUNDS_20020108"],
     testData[[]],
     testData[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],

     testData["FEDFUNDS_20020108"],
     "IV 2002 Without Lagged FEDFUNDS (All Quarters)"
     ),

    (results2002WithLaggedQ1,
     trainDataQ1[["FedFunds_2002_Lag1"]],
     trainDataQ1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],

     trainDataQ1["FEDFUNDS_20020108"],
     testDataQ1[["FedFunds_2002_Lag1"]],
     testDataQ1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],

     testDataQ1["FEDFUNDS_20020108"],
     "IV 2002 With Lagged FEDFUNDS (Q1)"
     ),

    (results2002WithLaggedAll,
     trainData[["FedFunds_2002_Lag1"]],
     trainData[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],

     trainData["FEDFUNDS_20020108"],
     testData[["FedFunds_2002_Lag1"]],
     testData[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"]],

     testData["FEDFUNDS_20020108"],
     "IV 2002 With Lagged FEDFUNDS (All Quarters)"
     )
]

allMetrics = []

for model, X_train, y_train, X_test, y_test, modelName in olsModels:
    allMetrics.append(calculateMetrics(model, X_train, y_train, X_test, y_test, modelName))

for model, X_trainExog, X_train_endog, y_train, X_testExog, X_test_endog, y_test, modelName in ivModels:
    allMetrics.append(calculateMetricsIV(model, X_trainExog, X_train_endog, y_train, X_testExog, X_test_endog, y_test, modelName))

MetricsDf = pd.DataFrame(allMetrics)
print("All Metrics OLS and IV")
print(MetricsDf)

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
        ["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"],
        "OLS 1997 Without Lag", i))

    resultsList.append(runOLSBootstrapOOB(mergedData, "FEDFUNDS_19970107",
        ["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"],
        "OLS 1997 With Lag", i))

    resultsList.append(runOLSBootstrapOOB(mergedData, "FEDFUNDS_20020108",
        ["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"],
        "OLS 2002 Without Lag", i))

    resultsList.append(runOLSBootstrapOOB(mergedData, "FEDFUNDS_20020108",
        ["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"],
        "OLS 2002 With Lag", i))

    resultsList.append(runIVBootstrapOOB(mergedData, "FEDFUNDS_19970107",
        [],
        ["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"],
        ["OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
         "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
         "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3",
         "Oil_Price_1997_Lag1", "Oil_Price_1997_Lag2", "Oil_Price_1997_Lag3",
         "Unemployment_1997_Lag1", "Unemployment_1997_Lag2", "Unemployment_1997_Lag3"],
        "IV 1997 Without Lag", i))

    resultsList.append(runIVBootstrapOOB(mergedData, "FEDFUNDS_19970107",
        ["FedFunds_1997_Lag1"],
        ["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122", "OILPRICE_19970303", "UNRATE_19970110"],
        ["OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
         "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
         "Houses_1997_Lag1", "Oil_Price_1997_Lag1", "Unemployment_1997_Lag1"],
        "IV 1997 With Lag", i))

    resultsList.append(runIVBootstrapOOB(mergedData, "FEDFUNDS_20020108",
        [],
        ["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"],
        ["OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
         "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
         "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3",
         "Oil_Price_2002_Lag1", "Oil_Price_2002_Lag2", "Oil_Price_2002_Lag3",
         "Unemployment_2002_Lag1", "Unemployment_2002_Lag2", "Unemployment_2002_Lag3"],
        "IV 2002 Without Lag", i))

    resultsList.append(runIVBootstrapOOB(mergedData, "FEDFUNDS_20020108",
        ["FedFunds_2002_Lag1"],
        ["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117", "OILPRICE_20020102", "UNRATE_20020104"],
        ["OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
         "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
         "Houses_2002_Lag1", "Oil_Price_2002_Lag1", "Unemployment_2002_Lag1"],
        "IV 2002 With Lag", i))

bootstrapDf = pd.DataFrame(resultsList)
bootstrapSummary = bootstrapDf.groupby("Model").agg(["mean", "std"])
print(bootstrapSummary)