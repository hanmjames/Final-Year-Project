import joblib
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor

rfParams = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'max_features': None,
    'random_state': 42
}

rfWithLag = joblib.load("rf_model_WithLag_1981_1996.pkl")
rfWithoutLag = joblib.load("rf_model_WithoutLag_1981_1996.pkl")

fedFunds = pd.read_csv("../Training_and_Testing_Data/FedfundsTest.csv")
inflation = pd.read_csv("../Training_and_Testing_Data/InflationTest.csv")
realGDP = pd.read_csv("../Training_and_Testing_Data/RealGDPTest.csv")
potGDP = pd.read_csv("../Training_and_Testing_Data/PotGDPTest.csv")

fedFunds['observation_date'] = pd.to_datetime(fedFunds['observation_date'])
inflation['observation_date'] = pd.to_datetime(inflation['observation_date'])
realGDP['observation_date'] = pd.to_datetime(realGDP['observation_date'])
potGDP['observation_date'] = pd.to_datetime(potGDP['observation_date'])

startDate, endDate = '2000-01-01', '2006-12-31'
for name, df in zip(['fedFunds', 'inflation', 'realGDP', 'potGDP'], [fedFunds, inflation, realGDP, potGDP]):
    globals()[name] = df[(df['observation_date'] >= startDate) & (df['observation_date'] <= endDate)]

inflation['InflationRate'] = ((inflation['GDPDEF'] / inflation['GDPDEF'].shift(4)) - 1) * 100
mergedGDP = realGDP.merge(potGDP, on='observation_date')
mergedGDP['OutputGap'] = 100 * (np.log(mergedGDP['GDPC1']) - np.log(mergedGDP['GDPPOT']))

testData = fedFunds.merge(inflation[['observation_date', 'InflationRate']], on='observation_date')
testData = testData.merge(mergedGDP[['observation_date', 'OutputGap']], on='observation_date')
testData['FedFundsLag1'] = testData['FEDFUNDS'].shift(1)
testData.dropna(inplace=True)

featuresWithLag = ['InflationRate', 'OutputGap', 'FedFundsLag1']
featuresWithoutLag = ['InflationRate', 'OutputGap']

xWithLag = testData[featuresWithLag]
xWithoutLag = testData[featuresWithoutLag]
yTest = testData['FEDFUNDS']

testData['Predicted_FedFunds_With_Lag'] = rfWithLag.predict(xWithLag)
testData['Predicted_FedFunds_Without_Lag'] = rfWithoutLag.predict(xWithoutLag)

def evaluate(y_true, y_pred, label, k):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return {
        'Model': label,
        'R²': r2,
        'Adj R²': adj_r2,
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE/Mean': mean_squared_error(y_true, y_pred) / y_true.mean()
    }

results = [
    evaluate(yTest, testData['Predicted_FedFunds_With_Lag'], "Random Forest **With** Lagged FedFunds", k=3),
    evaluate(yTest, testData['Predicted_FedFunds_Without_Lag'], "Random Forest **Without** Lagged FedFunds", k=2)
]

rfTestResults = pd.DataFrame(results)
print("\nTest Results (2007–2013):")
print(rfTestResults)

def computeVIF(df, features):
    vif = pd.DataFrame()
    vif['Feature'] = features
    vif['VIF'] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif.sort_values(by='VIF', ascending=False)

print("\nVIF (With Lag):")
print(computeVIF(testData, featuresWithLag))
print("\nVIF (Without Lag):")
print(computeVIF(testData, featuresWithoutLag))

print("\nBuilt-in Feature Importances (With Lag):")
print(pd.Series(rfWithLag.feature_importances_, index=featuresWithLag).sort_values(ascending=False))

print("\nBuilt-in Feature Importances (Without Lag):")
print(pd.Series(rfWithoutLag.feature_importances_, index=featuresWithoutLag).sort_values(ascending=False))

pfiWithLag = permutation_importance(rfWithLag, xWithLag, yTest, n_repeats=30, random_state=42, scoring='neg_mean_squared_error')
pfi_dfWithLag = pd.DataFrame({
    "Feature": xWithLag.columns,
    "Importance": -1 * pfiWithLag.importances_mean,
    "StdDev": pfiWithLag.importances_std
}).sort_values(by="Importance", ascending=False)
print("\nPermutation Importance (Test, With Lag):")
print(pfi_dfWithLag)

pfiWithoutLag = permutation_importance(rfWithoutLag, xWithoutLag, yTest, n_repeats=30, random_state=42, scoring='neg_mean_squared_error')
pfi_dfWithoutLag = pd.DataFrame({
    "Feature": xWithoutLag.columns,
    "Importance": -1 * pfiWithoutLag.importances_mean,
    "StdDev": pfiWithoutLag.importances_std
}).sort_values(by="Importance", ascending=False)
print("\nPermutation Importance (Test, Without Lag):")
print(pfi_dfWithoutLag)

def LOOAnalysis(model_base, x_test, y_test, label):
    baseline_pred = model_base.predict(x_test)
    baseline_r2 = r2_score(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    results = []

    for feature in x_test.columns:
        x_test_drop = x_test.copy()
        x_test_drop[feature] = x_test[feature].mean()
        pred_drop = model_base.predict(x_test_drop)

        r2_drop = baseline_r2 - r2_score(y_test, pred_drop)
        rmse_drop = np.sqrt(mean_squared_error(y_test, pred_drop)) - baseline_rmse
        mae_drop = mean_absolute_error(y_test, pred_drop) - baseline_mae

        results.append({
            'Feature': feature,
            'R² Drop': round(r2_drop, 4),
            'RMSE Drop': round(rmse_drop, 4),
            'MAE Drop': round(mae_drop, 4)
        })

    print(f"\nLOO Analysis for {label}:")
    return pd.DataFrame(results).sort_values(by='R² Drop', ascending=False)

looWithLag = LOOAnalysis(rfWithLag, xWithLag, yTest, "Random Forest **With** Lagged FedFunds")
print(looWithLag)

looWithoutLag = LOOAnalysis(rfWithoutLag, xWithoutLag, yTest, "Random Forest **Without** Lagged FedFunds")
print(looWithoutLag)