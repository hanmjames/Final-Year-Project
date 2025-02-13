# Data Processing
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Modelling
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

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

features = ['InflationRate', 'OutputGap', 'PresidentParty', 'SenateMajority', 'HouseMajority', 'InflationInteraction', 'OutputGapInteraction', 'FedFundsLag1']
x = mergedDataset[features]
y = mergedDataset['FEDFUNDS']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)

rfRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
rfRegressor.fit(xTrain, yTrain)

# Predict on test set
yPred = rfRegressor.predict(xTest)

# Calculate Evaluation Metrics
mae = mean_absolute_error(yTest, yPred)
rmse = np.sqrt(mean_squared_error(yTest, yPred))
r2 = r2_score(yTest, yPred)

# Print results
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Get feature importances
feature_importances = pd.Series(rfRegressor.feature_importances_, index=x.columns)

# Sort and plot
plt.figure(figsize=(10, 6))
feature_importances.sort_values().plot(kind='barh', color='skyblue')
plt.title("Feature Importance in Predicting Fed Funds Rate")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()