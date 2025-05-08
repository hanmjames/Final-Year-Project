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
from sklearn.utils import resample
from joblib import Parallel, delayed
from tqdm import tqdm

rfParams = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'max_features': None,
    'random_state': 42
}

fedfundsData = pd.read_csv("../Training_and_Testing_Data/FedfundsTotal.csv")
fedfundsData['observation_date'] = pd.to_datetime(fedfundsData['observation_date'], format='%d/%m/%Y')

inflationData = pd.read_csv("../Training_and_Testing_Data/InflationTotal.csv")
inflationData['observation_date'] = pd.to_datetime(inflationData['observation_date'], format='%d/%m/%Y')

realGDPData = pd.read_csv("../Training_and_Testing_Data/RealGDPTotal.csv")
realGDPData['observation_date'] = pd.to_datetime(realGDPData['observation_date'], format='%d/%m/%Y')

potGDPData = pd.read_csv("../Training_and_Testing_Data/PotentialGDPTotal.csv")
potGDPData['observation_date'] = pd.to_datetime(potGDPData['observation_date'], format='%d/%m/%Y')

partyData = pd.read_csv("../Training_and_Testing_Data/partyTotal.csv")
partyData['observation_date'] = pd.to_datetime(partyData['observation_date'], format='%d/%m/%Y')

mergedDataset = (fedfundsData.merge(inflationData, on='observation_date')
                              .merge(realGDPData, on='observation_date')
                              .merge(potGDPData, on='observation_date')
                              .merge(partyData, on='observation_date'))

mergedDataset['InflationRate'] = ((mergedDataset['GDPDEF'] / mergedDataset['GDPDEF'].shift(4) - 1) * 100)
mergedDataset['OutputGap'] = 100 * (np.log(mergedDataset['GDPC1']) - np.log(mergedDataset['GDPPOT']))
mergedDataset['FedFundsLag1'] = mergedDataset['FEDFUNDS'].shift(1)
mergedDataset['InflationInteraction'] = mergedDataset['InflationRate'] * mergedDataset['PresidentParty']
mergedDataset['OutputGapInteraction'] = mergedDataset['OutputGap'] * mergedDataset['PresidentParty']

mergedDataset.drop(columns=['GDPDEF', 'GDPPOT', 'GDPC1'], inplace=True)
mergedDataset.dropna(inplace=True)

mergedDataset = mergedDataset[(mergedDataset['observation_date'].dt.year >= 1981) &
                               (mergedDataset['observation_date'].dt.year <= 1996)].copy()

features_WithLag = ['InflationRate', 'OutputGap', 'FedFundsLag1']
features_WithoutLag = ['InflationRate', 'OutputGap']
features_WithII = ['InflationRate', 'OutputGap', 'PresidentParty', 'InflationInteraction', 'OutputGapInteraction', 'FedFundsLag1']
features_WithoutII = ['InflationRate', 'OutputGap', 'PresidentParty', 'OutputGapInteraction', 'FedFundsLag1']

x_WithLag = mergedDataset[features_WithLag]
y = mergedDataset['FEDFUNDS']
xTrain_lag, xTest_lag, yTrain_lag, yTest_lag = train_test_split(x_WithLag, y, test_size=0.2, shuffle=False, random_state=42)
rf_WithLag = RandomForestRegressor(**rfParams).fit(xTrain_lag, yTrain_lag)
joblib.dump(rf_WithLag, "rf_model_WithLag_1981_1996.pkl")

x_WithoutLag = mergedDataset[features_WithoutLag]
xTrain_woLag, xTest_woLag, yTrain_woLag, yTest_woLag = train_test_split(x_WithoutLag, y, test_size=0.2, shuffle=False, random_state=42)
rf_WithoutLag = RandomForestRegressor(**rfParams).fit(xTrain_woLag, yTrain_woLag)
joblib.dump(rf_WithoutLag, "rf_model_WithoutLag_1981_1996.pkl")

x_WithII = mergedDataset[features_WithII]
xTrain_ii, xTest_ii, yTrain_ii, yTest_ii = train_test_split(x_WithII, y, test_size=0.2, shuffle=False, random_state=42)
rf_WithII = RandomForestRegressor(**rfParams).fit(xTrain_ii, yTrain_ii)
joblib.dump(rf_WithII, "rf_model_with_ii_1981_1996.pkl")

x_WithoutII = mergedDataset[features_WithoutII]
xTrain_noii, xTest_noii, yTrain_noii, yTest_noii = train_test_split(x_WithoutII, y, test_size=0.2, shuffle=False, random_state=42)
rf_WithoutII = RandomForestRegressor(**rfParams).fit(xTrain_noii, yTrain_noii)
joblib.dump(rf_WithoutII, "rf_model_without_ii_1981_1996.pkl")

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

def LOOAnalysis(model, X_train, y_train, X_test, y_test, label):
    results = []
    baseline_r2 = r2_score(y_test, model.predict(X_test))
    baselineRMSE = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    baseline_mae = mean_absolute_error(y_test, model.predict(X_test))

    for feature in X_train.columns:
        X_train_drop = X_train.drop(columns=[feature])
        X_test_drop = X_test.drop(columns=[feature])
        rf = RandomForestRegressor(**rfParams)
        rf.fit(X_train_drop, y_train)
        y_pred = rf.predict(X_test_drop)

        r2_drop = baseline_r2 - r2_score(y_test, y_pred)
        rmse_drop = np.sqrt(mean_squared_error(y_test, y_pred)) - baselineRMSE
        mae_drop = mean_absolute_error(y_test, y_pred) - baseline_mae

        results.append({"Feature": feature, "R² Drop": r2_drop, "RMSE Drop": rmse_drop, "MAE Drop": mae_drop})

    df = pd.DataFrame(results).sort_values(by='R² Drop', ascending=False)
    print(f"\nLOO Analysis for {label}:\n", df)

def computeVIF(X, label):
    vifData = pd.DataFrame()
    vifData["Feature"] = X.columns
    vifData["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(f"\nVIF for {label}:\n", vifData.sort_values(by="VIF", ascending=False))

models = [
    ("Random Forest **With** Lagged FedFunds", rf_WithLag, xTrain_lag, yTrain_lag, xTest_lag, yTest_lag),
    ("Random Forest **Without** Lagged FedFunds", rf_WithoutLag, xTrain_woLag, yTrain_woLag, xTest_woLag, yTest_woLag),
    ("Random Forest **With** Interaction", rf_WithII, xTrain_ii, yTrain_ii, xTest_ii, yTest_ii),
    ("Random Forest **Without** Interaction", rf_WithoutII, xTrain_noii, yTrain_noii, xTest_noii, yTest_noii)
]

results = [calculateRFMetrics(model, xTr, yTr, xTe, yTe, name) for name, model, xTr, yTr, xTe, yTe in models]
metricsDf = pd.DataFrame(results)
print("\nPerformance Summary (1981–1996 Training):")
print(metricsDf)

def runRfOobBootstrap(x, y, i):
    xBoot, yBoot = resample(x, y, replace=True, random_state=i)
    rf = RandomForestRegressor(**rfParams, oob_score=True, bootstrap=True)
    rf.fit(xBoot, yBoot)
    oob_pred = rf.oob_prediction_
    oob_mask = ~np.isnan(oob_pred)

    if oob_mask.sum() == 0:
        return {'Iteration': i, 'RMSE': np.nan, 'MAE': np.nan}

    rmse = np.sqrt(mean_squared_error(yBoot[oob_mask], oob_pred[oob_mask]))
    mae = mean_absolute_error(yBoot[oob_mask], oob_pred[oob_mask])
    return {'Iteration': i, 'RMSE': rmse, 'MAE': mae}

print("Built-in Feature Importances (rf_WithLag):")
print(pd.Series(rf_WithLag.feature_importances_, index=xTrain_lag.columns).sort_values(ascending=False))

featureImportances_woLag = pd.Series(rf_WithoutLag.feature_importances_, index=xTrain_woLag.columns)
featureImportances_woLag = featureImportances_woLag.sort_values(ascending=False)
print("\nBuilt-in Feature Importances (rf_WithoutLag):")
print(featureImportances_woLag)

computeVIF(xTrain_lag, "With Lagged FedFunds")
computeVIF(xTrain_woLag, "Without Lagged FedFunds")

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

looWithLag = LOOAnalysis(rf_WithLag, xTest_lag, yTest_lag, "Random Forest **With** Lagged FedFunds")
print(looWithLag)

looWithoutLag = LOOAnalysis(rf_WithoutLag, xTest_woLag, yTest_woLag, "Random Forest **Without** Lagged FedFunds")
print(looWithoutLag)