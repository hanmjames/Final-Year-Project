import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

fedfundsData = pd.read_csv("../Training_and_Testing_Data/FedfundsTotal.csv")
inflationData = pd.read_csv("../Training_and_Testing_Data/InflationTotal.csv")
realGDPData = pd.read_csv("../Training_and_Testing_Data/RealGDPTotal.csv")
potGDPData = pd.read_csv("../Training_and_Testing_Data/PotentialGDPTotal.csv")
#Dem = 0, Rep = 1
partyData = pd.read_csv("../Training_and_Testing_Data/partyTotal.csv")
#Dem = 1, Rep = 0
partyRevData = pd.read_csv("../Training_and_Testing_Data/partyTotalReversed.csv")

mergedDataset = (fedfundsData.merge(inflationData, on='observation_date').merge(realGDPData, on='observation_date').merge(potGDPData, on='observation_date').merge(partyData, on='observation_date').merge(partyRevData, on='observation_date'))

mergedDataset['observation_date'] = pd.to_datetime(mergedDataset['observation_date'])
mergedDataset['observation_date'] = mergedDataset['observation_date'].dt.strftime('%Y-%d-%m')

mergedDataset['InflationRate'] = ((mergedDataset['GDPDEF'] / mergedDataset['GDPDEF'].shift(4) - 1) * 100)
mergedDataset['OutputGap'] = 100 * (np.log(mergedDataset['GDPC1']) - np.log(mergedDataset['GDPPOT']))
mergedDataset['FedFundsLag1'] = mergedDataset['FEDFUNDS'].shift(1)
mergedDataset['InflationInteraction'] = mergedDataset['InflationRate'] * mergedDataset['PresidentParty']
mergedDataset['OutputGapInteraction'] = mergedDataset['OutputGap'] * mergedDataset['PresidentParty']
mergedDataset['InflationInteractionRev'] = mergedDataset['InflationRate'] * mergedDataset['PresidentPartyRev']
mergedDataset['OutputGapInteractionRev'] = mergedDataset['OutputGap'] * mergedDataset['PresidentPartyRev']

mergedDataset.drop(columns=['GDPDEF', 'GDPPOT', 'GDPC1'], inplace=True)
mergedDataset.dropna(inplace=True)

#Rep = 1, Dem = 0

print(mergedDataset.head())

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

yRev= mergedDataset['FEDFUNDS']
xRev = mergedDataset[['InflationRate', 'OutputGap', 'PresidentPartyRev', 'InflationInteractionRev', 'OutputGapInteractionRev', 'FedFundsLag1']]
xRev = add_constant(xRev)
politicalOLSRev = OLS(yRev, xRev).fit()
print(politicalOLSRev.summary())