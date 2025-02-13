from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.api import OLS, add_constant
pd.set_option('display.max_columns', None)

fedfundsData = pd.read_csv("FedfundsTotal.csv")
inflationData = pd.read_csv("InflationTotal.csv")
realGDPData = pd.read_csv("RealGDPTotal.csv")
potGDPData = pd.read_csv("PotentialGDPTotal.csv")
partyData = pd.read_csv("PartyTotal.csv")

mergedDataset = (fedfundsData.merge(inflationData, on='observation_date').merge(realGDPData, on='observation_date').merge(potGDPData, on='observation_date').merge(partyData, on='observation_date'))

mergedDataset['observation_date'] = pd.to_datetime(mergedDataset['observation_date'])
mergedDataset['observation_date'] = mergedDataset['observation_date'].dt.strftime('%Y-%d-%m')

mergedDataset['InflationRate'] = ((mergedDataset['GDPDEF'] / mergedDataset['GDPDEF'].shift(4) - 1) * 100)
mergedDataset['OutputGap'] = 100 * (np.log(mergedDataset['GDPC1']) - np.log(mergedDataset['GDPPOT']))
mergedDataset['FedFundsLag1'] = mergedDataset['FEDFUNDS'].shift(1)
mergedDataset['InflationInteraction'] = mergedDataset['InflationRate'] * mergedDataset['PresidentParty']
mergedDataset['OutputGapInteraction'] = mergedDataset['OutputGap'] * mergedDataset['PresidentParty']

mergedDataset.drop(columns=['GDPDEF', 'GDPPOT', 'GDPC1'], inplace=True)
mergedDataset.dropna(inplace=True)

#Rep = 1, Dem = 0

print(mergedDataset.head())

#OLS Regression
y = mergedDataset['FEDFUNDS']
x = mergedDataset[['InflationRate', 'OutputGap', 'SenateMajority', 'HouseMajority', 'PresidentParty', 'FedFundsLag1']]
x = add_constant(x)
politicalOLS = OLS(y, x).fit()
print(politicalOLS.summary())

y2 = mergedDataset['FEDFUNDS']
x2 = mergedDataset[['InflationRate', 'OutputGap', 'PresidentParty', 'FedFundsLag1']]
x2 = add_constant(x2)
politicalOLS2 = OLS(y2, x2).fit()
print(politicalOLS2.summary())

y3 = mergedDataset['FEDFUNDS']
x3 = mergedDataset[['InflationRate', 'OutputGap', 'PresidentParty', 'InflationInteraction', 'OutputGapInteraction', 'FedFundsLag1']]
x3 = add_constant(x3)
politicalOLS3 = OLS(y3, x3).fit()
print(politicalOLS3.summary())

pred1 = politicalOLS.predict(x)
pred2 = politicalOLS2.predict(x2)
pred3 = politicalOLS3.predict(x3)
actual = mergedDataset['FEDFUNDS']

plt.figure(figsize=(10, 6))
plt.plot(mergedDataset['observation_date'], actual, label = "Actual FedFunds Values", color = "pink")
plt.plot(mergedDataset["observation_date"], pred1, label = "All Political Variables", color = "red")
plt.plot(mergedDataset["observation_date"], pred2, label = "Only Ruling Party Variable", color = "blue")
plt.plot(mergedDataset["observation_date"], pred3, label = "Ruling Party and 2 Interaction Variables", color = "purple")


plt.show()