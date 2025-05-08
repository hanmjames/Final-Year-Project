# import pandas as pd
# import numpy as np
# import joblib
# import statsmodels.api as sm
# from sklearn.model_selection import train_test_split
# from statsmodels.api import OLS, add_constant
# from linearmodels.iv import IV2SLS
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.utils import resample
# pd.set_option('display.max_columns', None)
#
# from RegressionAnalysis import ols_1997WithLagAll
# from RegressionAddnlVars import results1997WithLaggedAll
#
# realGDP = pd.read_csv(r"../Training_and_Testing_Data/RealGDPTotal.csv")
# realGDP['observation_date'] = pd.to_datetime(realGDP['observation_date'], format='%d/%m/%Y')
#
# potGDP = pd.read_csv(r"../Training_and_Testing_Data/PotentialGDPTotal.csv")
# potGDP['observation_date'] = pd.to_datetime(potGDP['observation_date'], format='%d/%m/%Y')
#
# inflation = pd.read_csv(r"../Training_and_Testing_Data/InflationTotal.csv")
# inflation['observation_date'] = pd.to_datetime(inflation['observation_date'], format='%d/%m/%Y')
#
# fedFunds = pd.read_csv(r"../Training_and_Testing_Data/FedfundsTotal.csv")
# fedFunds['observation_date'] = pd.to_datetime(fedFunds['observation_date'], format='%d/%m/%Y')
#
# mergedData = (
#     realGDP.merge(potGDP, on="observation_date")
#     .merge(inflation, on="observation_date")
#     .merge(fedFunds, on="observation_date")
# )
#
# mergedData['observation_date'] = pd.to_datetime(mergedData['observation_date'])
# mergedData['InflationRate'] = (
#     (mergedData['GDPDEF'] / mergedData['GDPDEF'].shift(4) - 1) * 100
# )
# mergedData['OutputGap'] = 100 * (
#     np.log(mergedData['GDPC1']) - np.log(mergedData['GDPPOT'])
# )
# mergedData['FedFunds_Lag1'] = mergedData['FEDFUNDS'].shift(1)
#
# mergedData['Inflation_Rate_1997'] = mergedData['InflationRate']
# mergedData['OutputGap_1997'] = mergedData['OutputGap']
# mergedData['FedFunds_1997_Lag1'] = mergedData['FedFunds_Lag1']
# mergedData['FEDFUNDS_19970107'] = mergedData['FEDFUNDS']
#
# scaler = joblib.load("../Model_Training/3AddnlVarsScaler.joblib")
# featureCols = joblib.load("../Model_Training/3AddnlVarsScalerColumns.joblib")
#
# requiredCols = ["FEDFUNDS_19970107", "FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]
# mergedData.dropna(subset=requiredCols, inplace=True)
# fullFeatures = pd.DataFrame(0, index=mergedData.index, columns=featureCols)
#
# for col in requiredCols:
#     fullFeatures[col] = mergedData[col]
#
# scaledFeatures = scaler.transform(fullFeatures)
#
# mergedData[requiredCols] = scaledFeatures[:, [featureCols.index(col) for col in requiredCols]]
# mergedData = mergedData[(mergedData['observation_date'] >= '1970-01-01') & (mergedData['observation_date'] <= '2013-12-31')]
#
# windowSize = 20
# startYear = 1981
#
# coeffList = []
# dates = []
#
# for i in range(windowSize, len(mergedData)):
#     window = mergedData.iloc[i - windowSize:i]
#     X = window[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]]
#     y = window["FEDFUNDS_19970107"]
#     X = add_constant(X)
#
#     model = OLS(y, X).fit()
#     coeffList.append(model.params)
#     dates.append(window["observation_date"].iloc[-1])
#
# coeffDfFinal = pd.DataFrame(coeffList)
# coeffDfFinal["observation_date"] = dates
#
# print(coeffDfFinal.head())
#
# contributions = pd.DataFrame()
# contributions['observation_date'] = mergedData['observation_date'].copy()
#
# # Extract coefficients
# coeffs = ols_1997WithLagAll.params
#
# # Compute contributions (coefficient × standardized value)
# contributions['Lagged_FedFunds_Contribution'] = coeffs['FedFunds_1997_Lag1'] * mergedData['FedFunds_1997_Lag1']
# contributions['Inflation_Rate_Contribution'] = coeffs['Inflation_Rate_1997'] * mergedData['Inflation_Rate_1997']
# contributions['Output_Gap_Contribution'] = coeffs['OutputGap_1997'] * mergedData['OutputGap_1997']
# contributions['Total_Predicted'] = (
#     coeffs['const']
#     + contributions['Lagged_FedFunds_Contribution']
#     + contributions['Inflation_Rate_Contribution']
#     + contributions['Output_Gap_Contribution']
# )
#
# # # Plotting contributions
# # plt.figure(figsize=(14, 6))
# # plt.plot(contributions['observation_date'], contributions['Lagged_FedFunds_Contribution'], label='Lagged Fed Funds Contribution', color='blue')
# # plt.plot(contributions['observation_date'], contributions['Inflation_Rate_Contribution'], label='Inflation Rate Contribution', color='orange')
# # plt.plot(contributions['observation_date'], contributions['Output_Gap_Contribution'], label='Output Gap Contribution', color='green')
# # plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
# # plt.title('Variable Contributions Over Time (OLS 1997 With Lagged FedFunds)')
# # plt.xlabel('Date')
# # plt.ylabel('Contribution to Predicted Fed Funds Rate')
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()
#
# # Filter to start plotting from 1981
# contributions = contributions[contributions['observation_date'] >= '1981-01-01']
#
# # Plotting contributions
# plt.figure(figsize=(14, 6))
# plt.plot(contributions['observation_date'], contributions['Lagged_FedFunds_Contribution'], label='Lagged Fed Funds Contribution', color='blue')
# plt.plot(contributions['observation_date'], contributions['Inflation_Rate_Contribution'], label='Inflation Rate Contribution', color='orange')
# plt.plot(contributions['observation_date'], contributions['Output_Gap_Contribution'], label='Output Gap Contribution', color='green')
# plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
# plt.title('Variable Contributions Over Time (OLS 1997 With Lagged FedFunds)')
# plt.xlabel('Date')
# plt.ylabel('Contribution to Predicted Fed Funds Rate')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
#
# # # Prepare contribution DataFrame
# # ivContributions = pd.DataFrame()
# # ivContributions['observation_date'] = mergedData['observation_date'].copy()
# #
# # # Extract relevant coefficients from the IV model
# # ivCoeffs = results1997WithLaggedAll.params
# #
# # # Compute contributions (coeff × standardized feature value)
# # ivContributions['Lagged_FedFunds_Contribution'] = ivCoeffs['FedFunds_1997_Lag1'] * mergedData['FedFunds_1997_Lag1']
# # ivContributions['Inflation_Rate_Contribution'] = ivCoeffs['Inflation_Rate_1997'] * mergedData['Inflation_Rate_1997']
# # ivContributions['Output_Gap_Contribution'] = ivCoeffs['OutputGap_1997'] * mergedData['OutputGap_1997']
# # ivContributions['Total_Predicted'] = (
# #     ivCoeffs['const']
# #     + ivContributions['Lagged_FedFunds_Contribution']
# #     + ivContributions['Inflation_Rate_Contribution']
# #     + ivContributions['Output_Gap_Contribution']
# # )
# #
# # # Plotting contributions from IV model
# # plt.figure(figsize=(14, 6))
# # plt.plot(ivContributions['observation_date'], ivContributions['Lagged_FedFunds_Contribution'], label='Lagged Fed Funds Contribution', color='blue')
# # plt.plot(ivContributions['observation_date'], ivContributions['Inflation_Rate_Contribution'], label='Inflation Rate Contribution', color='orange')
# # plt.plot(ivContributions['observation_date'], ivContributions['Output_Gap_Contribution'], label='Output Gap Contribution', color='green')
# # plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
# # plt.title('Variable Contributions Over Time (IV 1997 With Lagged FedFunds)')
# # plt.xlabel('Date')
# # plt.ylabel('Contribution to Predicted Fed Funds Rate')
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant

# Load and preprocess data
realGDP = pd.read_csv("../Training_and_Testing_Data/RealGDPTotal.csv")
potGDP = pd.read_csv("../Training_and_Testing_Data/PotentialGDPTotal.csv")
inflation = pd.read_csv("../Training_and_Testing_Data/InflationTotal.csv")
fedFunds = pd.read_csv("../Training_and_Testing_Data/FedfundsTotal.csv")

for df in [realGDP, potGDP, inflation, fedFunds]:
    df['observation_date'] = pd.to_datetime(df['observation_date'], format='%d/%m/%Y')

mergedData = (
    realGDP.merge(potGDP, on="observation_date")
           .merge(inflation, on="observation_date")
           .merge(fedFunds, on="observation_date")
)

# Feature engineering
mergedData['InflationRate'] = (mergedData['GDPDEF'] / mergedData['GDPDEF'].shift(4) - 1) * 100
mergedData['OutputGap'] = 100 * (np.log(mergedData['GDPC1']) - np.log(mergedData['GDPPOT']))
mergedData['FedFunds_Lag1'] = mergedData['FEDFUNDS'].shift(1)

# Assign model-consistent variable names
mergedData['Inflation_Rate_1997'] = mergedData['InflationRate']
mergedData['OutputGap_1997'] = mergedData['OutputGap']
mergedData['FedFunds_1997_Lag1'] = mergedData['FedFunds_Lag1']
mergedData['FEDFUNDS_19970107'] = mergedData['FEDFUNDS']

# Load 1997 scaler and transform relevant columns
scaler = joblib.load("../Model_Training/3AddnlVarsScaler.joblib")
featureCols = joblib.load("../Model_Training/3AddnlVarsScalerColumns.joblib")
requiredCols = ["FEDFUNDS_19970107", "FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]

# Fill zeros for missing model features, then overwrite the ones we care about
fullFeatures = pd.DataFrame(0, index=mergedData.index, columns=featureCols)
for col in requiredCols:
    fullFeatures[col] = mergedData[col]

scaled = scaler.transform(fullFeatures)
mergedData[requiredCols] = scaled[:, [featureCols.index(col) for col in requiredCols]]

# Filter valid range
mergedData.dropna(subset=requiredCols, inplace=True)
mergedData = mergedData[(mergedData['observation_date'] >= '1970-01-01') & (mergedData['observation_date'] <= '2013-12-31')]

# Rolling OLS setup
windowSize = 20
coeffList = []
dates = []

for i in range(windowSize, len(mergedData)):
    window = mergedData.iloc[i - windowSize:i]
    X = window[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]]
    y = window["FEDFUNDS_19970107"]
    X = add_constant(X)

    model = OLS(y, X).fit()
    coeffList.append(model.params)
    dates.append(window["observation_date"].iloc[-1])

# Construct DataFrame and plot
coeffDf = pd.DataFrame(coeffList)
coeffDf["observation_date"] = dates
coeffDf = coeffDf[coeffDf["observation_date"] >= "1981-01-01"]

plt.figure(figsize=(14, 6))
plt.plot(coeffDf["observation_date"], coeffDf["FedFunds_1997_Lag1"], label="Lagged Fed Funds", color="blue")
plt.plot(coeffDf["observation_date"], coeffDf["Inflation_Rate_1997"], label="Inflation Rate", color="orange")
plt.plot(coeffDf["observation_date"], coeffDf["OutputGap_1997"], label="Output Gap", color="green")
plt.axhline(0, linestyle='--', color='black', linewidth=0.7)
plt.title("Rolling OLS Coefficients (OLS 1997 With Lagged FedFunds)")
plt.xlabel("Date")
plt.ylabel("Coefficient Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from linearmodels.iv import IV2SLS

# Create instrument lags
for i in range(1, 4):
    mergedData[f'InflationRate_Lag{i}'] = mergedData['InflationRate'].shift(i)
    mergedData[f'OutputGap_Lag{i}'] = mergedData['OutputGap'].shift(i)

# Drop rows with any missing values from required + instrument columns
ivCols = requiredCols + [f'InflationRate_Lag{i}' for i in range(1, 4)] + [f'OutputGap_Lag{i}' for i in range(1, 4)]
mergedData.dropna(subset=ivCols, inplace=True)

# Rolling IV setup
ivCoeffList = []
ivDates = []

for i in range(windowSize, len(mergedData)):
    window = mergedData.iloc[i - windowSize:i]

    y = window["FEDFUNDS_19970107"]
    exog = add_constant(window[["FedFunds_1997_Lag1"]])
    endog = window[["Inflation_Rate_1997", "OutputGap_1997"]]
    instruments = window[["InflationRate_Lag1", "InflationRate_Lag2", "InflationRate_Lag3",
                          "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]]

    try:
        model = IV2SLS(dependent=y, exog=exog, endog=endog, instruments=instruments).fit()
        coeffs = model.params[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]]
        ivCoeffList.append(coeffs)
        ivDates.append(window["observation_date"].iloc[-1])
    except Exception:
        continue

# Construct DataFrame and plot
ivCoeffDf = pd.DataFrame(ivCoeffList)
ivCoeffDf["observation_date"] = ivDates
ivCoeffDf = ivCoeffDf[ivCoeffDf["observation_date"] >= "1981-01-01"]

plt.figure(figsize=(14, 6))
plt.plot(ivCoeffDf["observation_date"], ivCoeffDf["FedFunds_1997_Lag1"], label="Lagged Fed Funds", color="blue")
plt.plot(ivCoeffDf["observation_date"], ivCoeffDf["Inflation_Rate_1997"], label="Inflation Rate", color="orange")
plt.plot(ivCoeffDf["observation_date"], ivCoeffDf["OutputGap_1997"], label="Output Gap", color="green")
plt.axhline(0, linestyle='--', color='black', linewidth=0.7)
plt.title("Rolling IV Coefficients (IV 1997 With Lagged FedFunds)")
plt.xlabel("Date")
plt.ylabel("Coefficient Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
