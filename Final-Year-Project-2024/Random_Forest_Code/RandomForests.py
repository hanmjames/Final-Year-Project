import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from matplotlib import pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.utils import resample
from tqdm import tqdm

fedfundsData = pd.read_csv("../Training_and_Testing_Data/FedfundsTotal.csv")
inflationData = pd.read_csv("../Training_and_Testing_Data/InflationTotal.csv")
realGDPData = pd.read_csv("../Training_and_Testing_Data/RealGDPTotal.csv")
potGDPData = pd.read_csv("../Training_and_Testing_Data/PotentialGDPTotal.csv")
partyData = pd.read_csv("../Training_and_Testing_Data/partyTotal.csv")

mergedDataset = (fedfundsData.merge(inflationData, on='observation_date')
                              .merge(realGDPData, on='observation_date')
                              .merge(potGDPData, on='observation_date')
                              .merge(partyData, on='observation_date'))

mergedDataset['observation_date'] = pd.to_datetime(mergedDataset['observation_date'])
mergedDataset['InflationRate'] = ((mergedDataset['GDPDEF'] / mergedDataset['GDPDEF'].shift(4) - 1) * 100)
mergedDataset['OutputGap'] = 100 * (np.log(mergedDataset['GDPC1']) - np.log(mergedDataset['GDPPOT']))
mergedDataset['FedFundsLag1'] = mergedDataset['FEDFUNDS'].shift(1)
mergedDataset['InflationInteraction'] = mergedDataset['InflationRate'] * mergedDataset['PresidentParty']
mergedDataset['OutputGapInteraction'] = mergedDataset['OutputGap'] * mergedDataset['PresidentParty']

mergedDataset.drop(columns=['GDPDEF', 'GDPPOT', 'GDPC1'], inplace=True)
mergedDataset.dropna(inplace=True)

features = ['InflationRate', 'OutputGap', 'PresidentParty',
            'InflationInteraction', 'OutputGapInteraction', 'FedFundsLag1']
x = mergedDataset[features]
y = mergedDataset['FEDFUNDS']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)

rfRegressor = RandomForestRegressor(
    max_depth=13,
    min_samples_leaf=4,
    min_samples_split=3,
    n_estimators=397,
    random_state=42
)
rfRegressor.fit(xTrain, yTrain)

yPred = rfRegressor.predict(xTest)
baseline_mae = mean_absolute_error(yTest, yPred)
baselineRMSE = np.sqrt(mean_squared_error(yTest, yPred))
baseline_r2 = r2_score(yTest, yPred)

print(f"Mean Absolute Error (MAE): {baseline_mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {baselineRMSE:.4f}")
print(f"R² Score: {baseline_r2:.4f}")

result = permutation_importance(rfRegressor, xTest, yTest, n_repeats=30, random_state=42, scoring='r2')

importancesDf = pd.DataFrame({
    'Variable': xTest.columns,
    'Importance': result.importances_mean,
    'StdDev': result.importances_std
}).sort_values(by='Importance', ascending=False)

print("\nPermutation Feature Importance:")
print(importancesDf)

plt.figure(figsize=(10, 6))
plt.barh(importancesDf['Variable'], importancesDf['Importance'], xerr=importancesDf['StdDev'], color='skyblue')
plt.xlabel('Importance (Decrease in R²)')
plt.ylabel('Feature')
plt.title('Permutation Feature Importance')
plt.show()

results = []
for feature in features:
    xTrainDrop = xTrain.drop(columns=[feature])
    xTestDrop = xTest.drop(columns=[feature])
    rf = RandomForestRegressor(n_estimators=397, max_depth=13, min_samples_leaf=4, min_samples_split=3, random_state=42)
    rf.fit(xTrainDrop, yTrain)
    yPred = rf.predict(xTestDrop)
    mae = mean_absolute_error(yTest, yPred)
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    r2 = r2_score(yTest, yPred)
    mae_drop = mae - baseline_mae
    rmse_drop = rmse - baselineRMSE
    r2_drop = baseline_r2 - r2

    results.append({
        'Feature': feature,
        'MAE Drop': mae_drop,
        'RMSE Drop': rmse_drop,
        'R² Drop': r2_drop
    })

looResults = pd.DataFrame(results).sort_values(by='R² Drop', ascending=False)
print("\nLeave-One-Out Analysis:")
print(looResults)

def noiseSensitivityAnalysis(x_train, x_test, y_train, y_test, noise_levels=[0.01, 0.05, 0.1, 0.2]):
    results = []
    rf = RandomForestRegressor(n_estimators=397, max_depth=13, min_samples_leaf=4, min_samples_split=3, random_state=42)
    rf.fit(x_train, y_train)
    baselineRMSE = np.sqrt(mean_squared_error(y_test, rf.predict(x_test)))
    
    for noise in noise_levels:
        x_test_noisy = x_test + np.random.normal(loc=0, scale=noise, size=x_test.shape)
        y_pred = rf.predict(x_test_noisy)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmseInSamplecrease = ((rmse - baselineRMSE) / baselineRMSE) * 100  

        results.append({
            'Noise Level': noise,
            'RMSE': rmse,
            'RMSE Increase (%)': rmseInSamplecrease
        })

    return pd.DataFrame(results)

noiseResults = noiseSensitivityAnalysis(xTrain, xTest, yTrain, yTest)
print(noiseResults)

'''
Unexpectedly, performance improved slightly with low noise, suggesting possible overfitting to the original data. Small noise may act as regularization, improving generalization.
Higher noise starts to degrade performance, showing the model is sensitive to moderate disturbances.
Performance drop is smaller than at 0.10, suggesting the model may be less sensitive to larger noise levels due to random forest ensemble effects.
'''

vifData = pd.DataFrame()
vifData["Feature"] = x.columns
vifData["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(vifData.sort_values(by="VIF", ascending=False))
'''
VIF (Variance Inflation Factor) measures how much the variance of a regression coefficient is inflated due to multicollinearity with other predictors; a VIF > 5 indicates high multicollinearity.
'''

features2 = ['InflationRate', 'OutputGap', 'PresidentParty', 'OutputGapInteraction', 'FedFundsLag1']
x2 = mergedDataset[features2]
y2 = mergedDataset['FEDFUNDS']

xTrain2, xTest2, yTrain2, yTest2 = train_test_split(x2, y2, test_size=0.2, random_state=42, shuffle=False)

rfRegressor2 = RandomForestRegressor(max_depth=13,min_samples_leaf=4,min_samples_split=3,n_estimators=397,random_state=42)
rfRegressor2.fit(xTrain2, yTrain2)

yPred2 = rfRegressor2.predict(xTest2)
baseline_mae2 = mean_absolute_error(yTest2, yPred2)
baselineRMSE2 = np.sqrt(mean_squared_error(yTest2, yPred2))
baseline_r22 = r2_score(yTest2, yPred2)

print(f"Mean Absolute Error (MAE): {baseline_mae2:.4f}")
print(f"Root Mean Squared Error (RMSE): {baselineRMSE2:.4f}")
print(f"R² Score: {baseline_r22:.4f}")

result2 = permutation_importance(rfRegressor2, xTest2, yTest2, n_repeats=30, random_state=42, scoring='r2')

importancesDf2 = pd.DataFrame({
    'Variable': xTest2.columns,
    'Importance': result2.importances_mean,
    'StdDev': result2.importances_std
}).sort_values(by='Importance', ascending=False)

print("\nPermutation Feature Importance:")
print(importancesDf2)

plt.figure(figsize=(10, 6))
plt.barh(importancesDf2['Variable'], importancesDf2['Importance'], xerr=importancesDf2['StdDev'], color='skyblue')
plt.xlabel('Importance (Decrease in R²) Without II')
plt.ylabel('Feature Without II')
plt.title('Permutation Feature Importance Without II')
plt.show()

results2 = []
for feature in features2:

    xTrainDrop2 = xTrain2.drop(columns=[feature])
    xTestDrop2 = xTest2.drop(columns=[feature])

    rf = RandomForestRegressor(n_estimators=397, max_depth=13,
                               min_samples_leaf=4, min_samples_split=3, random_state=42)
    rf.fit(xTrainDrop2, yTrain2)

    yPred2 = rf.predict(xTestDrop2)
    mae2 = mean_absolute_error(yTest2, yPred2)
    rmse2 = np.sqrt(mean_squared_error(yTest2, yPred2))
    r22 = r2_score(yTest2, yPred2)

    mae_drop2 = mae2 - baseline_mae
    rmse_drop2 = rmse2 - baselineRMSE
    r2_drop2 = baseline_r2 - r22

    results2.append({
        'Feature': feature,
        'MAE Drop': mae_drop2,
        'RMSE Drop': rmse_drop2,
        'R² Drop': r2_drop2
    })

looResults2 = pd.DataFrame(results2).sort_values(by='R² Drop', ascending=False)
print("\nLeave-One-Out Analysis:")
print(looResults2)

noiseResults2 = noiseSensitivityAnalysis(xTrain2, xTest2, yTrain2, yTest2)
print(noiseResults2)

vifData2 = pd.DataFrame()
vifData2["Feature"] = x2.columns
vifData2["VIF"] = [variance_inflation_factor(x2.values, i) for i in range(x2.shape[1])]
print(vifData2.sort_values(by="VIF", ascending=False))

rfRegressor = RandomForestRegressor(n_estimators=397,  max_depth=13,min_samples_leaf=4,min_samples_split=3,oob_score=True,  random_state=42)

rfRegressor.fit(x2, y2)

oobr2 = rfRegressor.oob_score
oobpredictions = rfRegressor.oob_prediction_

oobmae = mean_absolute_error(y2, oobpredictions)
oobRMSE = np.sqrt(mean_squared_error(y2, oobpredictions))

print(f"Out-of-Bag R²: {oobr2:.4f}")
print(f"Out-of-Bag MAE: {oobmae:.4f}")
print(f"Out-of-Bag RMSE: {oobRMSE:.4f}")

loo = LeaveOneOut()
rf = RandomForestRegressor(n_estimators=50,max_depth=13,min_samples_leaf=4,min_samples_split=3,random_state=42)

mseValues = []

for trainIndex, testIndex in loo.split(x2):
    xTrain, xTest = x2.iloc[trainIndex], x2.iloc[testIndex]
    yTrain, yTest = y2.iloc[trainIndex], y2.iloc[testIndex]

    rf.fit(xTrain, yTrain)
    yPred = rf.predict(xTest)
    mseValues.append(mean_squared_error(yTest, yPred))

print(f"LOOCV Mean Squared Error: {np.mean(mseValues):.4f}")
print(f"LOOCV RMSE: {np.sqrt(np.mean(mseValues)):.4f}")

'''
1. Permutation Feature Importance

    Process: Randomly shuffle the values of one variable while keeping all others unchanged.
    Effect: Measures how much the model’s performance (e.g., R² or RMSE) drops due to the disruption.
    Interpretation: A larger drop means the variable is crucial; a small or negative drop means it's less helpful or even harmful.
    Key Insight: This shows the marginal impact of a variable while accounting for interactions with other predictors.'''

'''
2. Leave-One-Out Analysis (LOO or Drop-One-Out)

    Process: Completely remove one variable from the dataset and re-train the model from scratch.
    Effect: Compare the model’s performance without that variable against the baseline with all variables.
    Interpretation: A large performance drop indicates that the variable is essential. A small drop or performance improvement means the variable isn’t adding value.
🔎 Key Insight: This shows the independent contribution of a variable, isolated from other predictors.
'''

'''
Output Gap does have rpedictive reelvance but may be introducing noise due to multicollinearity or interaction w other variables'''

joblib.dump(rfRegressor, "rf_model_with_ii.pkl")
joblib.dump(rfRegressor2, "rf_model_without_ii.pkl")

features_WithLag = ['InflationRate', 'OutputGap', 'FedFundsLag1']
x_WithLag = mergedDataset[features_WithLag].copy()
y_WithLag = mergedDataset['FEDFUNDS']

rf_WithLag = RandomForestRegressor(n_estimators=397,max_depth=13,min_samples_leaf=4,min_samples_split=3,random_state=42)
rf_WithLag.fit(x_WithLag, y_WithLag)
joblib.dump(rf_WithLag, "rf_model_WithLag.pkl")

features_WithoutLag = ['InflationRate', 'OutputGap']
x_WithoutLag = mergedDataset[features_WithoutLag].copy()
y_WithoutLag = mergedDataset['FEDFUNDS']

rf_WithoutLag = RandomForestRegressor(n_estimators=397,max_depth=13,min_samples_leaf=4,min_samples_split=3,random_state=42)
rf_WithoutLag.fit(x_WithoutLag, y_WithoutLag)
joblib.dump(rf_WithoutLag, "rf_model_WithoutLag.pkl")

def calculateRFMetrics(model, X_train, y_train, X_test, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    r2InSample = r2_score(y_train, y_train_pred)
    rmseInSample = np.sqrt(mean_squared_error(y_train, y_train_pred))
    maeInSample = mean_absolute_error(y_train, y_train_pred)
    mse_meanInSample = mean_squared_error(y_train, y_train_pred) / y_train.mean()
    adj_r2InSample = 1 - (1 - r2InSample) * ((len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))

    r2OOS = r2_score(y_test, y_test_pred)
    rmseOOS = np.sqrt(mean_squared_error(y_test, y_test_pred))
    maeOOS = mean_absolute_error(y_test, y_test_pred)
    mse_meanOOS = mean_squared_error(y_test, y_test_pred) / y_test.mean()

    return {
        "Model": model_name,
        "In-sample R-squared": r2InSample,
        "In-sample RMSE": rmseInSample,
        "In-sample MAE": maeInSample,
        "Out-of-sample RMSE": rmseOOS,
        "Out-of-sample MAE": maeOOS,
        "Adjusted R-squared": adj_r2InSample,
        "MSE/Mean (In-sample)": mse_meanInSample,
        "MSE/Mean (Out-of-sample)": mse_meanOOS
    }

xTrain_lag, xTest_lag, yTrain_lag, yTest_lag = train_test_split(x_WithLag, y_WithLag, test_size=0.2, shuffle=False, random_state=42)

xTrain_WithoutLag, xTest_WithoutLag, yTrain_WithoutLag, yTest_WithoutLag = train_test_split(x_WithoutLag, y_WithoutLag, test_size=0.2, shuffle=False, random_state=42)

results = []

results.append(calculateRFMetrics(
    rf_WithLag, xTrain_lag, yTrain_lag, xTest_lag, yTest_lag,
    "Random Forest **With** Lagged FedFunds"
))

results.append(calculateRFMetrics(
    rf_WithoutLag, xTrain_WithoutLag, yTrain_WithoutLag, xTest_WithoutLag, yTest_WithoutLag,
    "Random Forest **Without** Lagged FedFunds"
))

metricsDf = pd.DataFrame(results)
print(metricsDf)

def computeVIF(X, label):
    vifData = pd.DataFrame()
    vifData["Feature"] = X.columns
    vifData["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(f"\nVIF for {label}:\n", vifData.sort_values(by="VIF", ascending=False))

computeVIF(x_WithLag, "With Lagged FedFunds")
computeVIF(x_WithoutLag, "Without Lagged FedFunds")

def permImportance(model, X_test, y_test, label):
    result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring='r2')
    importancesDf = pd.DataFrame({
        'Variable': X_test.columns,
        'Importance': result.importances_mean,
        'StdDev': result.importances_std
    }).sort_values(by='Importance', ascending=False)

    print(f"\nPermutation Importance for {label}:\n", importancesDf)

permImportance(rf_WithLag, xTest_lag, yTest_lag, "With Lagged FedFunds")
permImportance(rf_WithoutLag, xTest_WithoutLag, yTest_WithoutLag, "Without Lagged FedFunds")

def LOOAnalysis(model, X_train, y_train, X_test, y_test, label):
    results = []
    baseline_r2 = r2_score(y_test, model.predict(X_test))
    baselineRMSE = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    baseline_mae = mean_absolute_error(y_test, model.predict(X_test))

    for feature in X_train.columns:
        X_train_drop = X_train.drop(columns=[feature])
        X_test_drop = X_test.drop(columns=[feature])

        rf = RandomForestRegressor(n_estimators=397, max_depth=13,min_samples_leaf=4, min_samples_split=3, random_state=42)
        rf.fit(X_train_drop, y_train)
        y_pred = rf.predict(X_test_drop)

        r2_drop = baseline_r2 - r2_score(y_test, y_pred)
        rmse_drop = np.sqrt(mean_squared_error(y_test, y_pred)) - baselineRMSE
        mae_drop = mean_absolute_error(y_test, y_pred) - baseline_mae

        results.append({
            "Feature": feature,
            "R² Drop": r2_drop,
            "RMSE Drop": rmse_drop,
            "MAE Drop": mae_drop
        })

    df = pd.DataFrame(results).sort_values(by='R² Drop', ascending=False)
    print(f"\nLOO Analysis for {label}:\n", df)

LOOAnalysis(rf_WithLag, xTrain_lag, yTrain_lag, xTest_lag, yTest_lag, "With Lagged FedFunds")
LOOAnalysis(rf_WithoutLag, xTrain_WithoutLag, yTrain_WithoutLag, xTest_WithoutLag, yTest_WithoutLag, "Without Lagged FedFunds")

rf_WithLagOOB = RandomForestRegressor(n_estimators=397,max_depth=13,min_samples_leaf=4,min_samples_split=3,oob_score=True,bootstrap=True,random_state=42)
rf_WithLagOOB.fit(x_WithLag, y_WithLag)

oobr2_with = rf_WithLagOOB.oob_score
oobpred_with = rf_WithLagOOB.oob_prediction_
oobmae_with = mean_absolute_error(y_WithLag, oobpred_with)
oobRMSE_with = np.sqrt(mean_squared_error(y_WithLag, oobpred_with))

print("OOB Results (With Lagged FedFunds):")
print(f"R²: {oobr2_with:.4f}")
print(f"MAE: {oobmae_with:.4f}")
print(f"RMSE: {oobRMSE_with:.4f}")

rf_WithoutLagOOB = RandomForestRegressor(n_estimators=397,max_depth=13,min_samples_leaf=4,min_samples_split=3, oob_score=True,bootstrap=True,random_state=42)
rf_WithoutLagOOB.fit(x_WithoutLag, y_WithoutLag)

oobr2_without = rf_WithoutLagOOB.oob_score
oobpred_without = rf_WithoutLagOOB.oob_prediction_
oobmae_without = mean_absolute_error(y_WithoutLag, oobpred_without)
oobRMSE_without = np.sqrt(mean_squared_error(y_WithoutLag, oobpred_without))

print("\nOOB Results (Without Lagged FedFunds):")
print(f"R²: {oobr2_without:.4f}")
print(f"MAE: {oobmae_without:.4f}")
print(f"RMSE: {oobRMSE_without:.4f}")

nIterations = 1000
bootstrapResults_WithoutLag = []

for i in tqdm(range(nIterations), desc="Bootstrapping Without Lag"):
    xBoot_WithoutLag, yBoot_WithoutLag = resample(x_WithoutLag, y_WithoutLag, replace=True, random_state=i)

    rf_WithoutLag = RandomForestRegressor(n_estimators=397,max_depth=13,min_samples_leaf=4,min_samples_split=3,random_state=42)
    rf_WithoutLag.fit(xBoot_WithoutLag, yBoot_WithoutLag)
    yPred_WithoutLag = rf_WithoutLag.predict(xTest_WithoutLag)

    bootstrapResults_WithoutLag.append({
        'Iteration': i,
        'R2': r2_score(yTest_WithoutLag, yPred_WithoutLag),
        'RMSE': np.sqrt(mean_squared_error(yTest_WithoutLag, yPred_WithoutLag)),
        'MAE': mean_absolute_error(yTest_WithoutLag, yPred_WithoutLag)
    })

bootstrapDf_WithoutLag = pd.DataFrame(bootstrapResults_WithoutLag)
summary_WithoutLag = bootstrapDf_WithoutLag[['R2', 'RMSE', 'MAE']].agg(['mean', 'std'])
print("\nBootstrap Summary (Without Lagged FedFunds):")
print(summary_WithoutLag)