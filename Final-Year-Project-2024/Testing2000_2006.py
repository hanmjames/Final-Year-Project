import pandas as pd
import numpy as np
from statsmodels.api import add_constant
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

from RegressionAnalysis import (
    # OLS Models
    ols_1997_without_lag_q1, ols_1997_without_lag_all,
    ols_1997_with_lag_q1, ols_1997_with_lag_all,
    ols_2002_without_lag_q1, ols_2002_without_lag_all,
    ols_2002_with_lag_q1, ols_2002_with_lag_all,

    # IV Models
    results_1997_without_lagged_q1, results_1997_without_lagged_all,
    results_1997_with_lagged_q1, results_1997_with_lagged_all,
    results_2002_without_lagged_q1, results_2002_without_lagged_all,
    results_2002_with_lagged_q1, results_2002_with_lagged_all,

    merged_data
)

fedFundsTest = pd.read_csv("FedfundsTest.csv")
inflationTest = pd.read_csv("InflationTest.csv")
realGDPTest = pd.read_csv("RealGDPTest.csv")
potGDPTest = pd.read_csv("PotGDPTest.csv")

fedFundsTest['observation_date'] = pd.to_datetime(fedFundsTest['observation_date'])
inflationTest['observation_date'] = pd.to_datetime(inflationTest['observation_date'])
realGDPTest['observation_date'] = pd.to_datetime(realGDPTest['observation_date'])
potGDPTest['observation_date'] = pd.to_datetime(potGDPTest['observation_date'])

start_date = '1998-01-01'
end_date = '2006-12-31'

fedFundsTest = fedFundsTest[(fedFundsTest['observation_date'] >= start_date) & (fedFundsTest['observation_date'] <= end_date)]
inflationTest = inflationTest[(inflationTest['observation_date'] >= start_date) & (inflationTest['observation_date'] <= end_date)]
realGDPTest = realGDPTest[(realGDPTest['observation_date'] >= start_date) & (realGDPTest['observation_date'] <= end_date)]
potGDPTest = potGDPTest[(potGDPTest['observation_date'] >= start_date) & (potGDPTest['observation_date'] <= end_date)]

inflationTest['Inflation_Rate'] = (
    (inflationTest['GDPDEF'] / inflationTest['GDPDEF'].shift(4) - 1) * 100
)

merged_pot_gdp = potGDPTest[['observation_date', 'GDPPOT']].merge(
    realGDPTest[['observation_date', 'GDPC1']], on='observation_date'
)

merged_pot_gdp['OutputGap'] = 100 * (
    (np.log(merged_pot_gdp['GDPC1']) - np.log(merged_pot_gdp['GDPPOT']))
)

for lag in range(1, 4):
    fedFundsTest[f'FEDFUNDS_Lag{lag}'] = fedFundsTest['FEDFUNDS'].shift(lag)
    inflationTest[f'Inflation_Rate_Lag{lag}'] = inflationTest['Inflation_Rate'].shift(lag)
    merged_pot_gdp[f'OutputGap_Lag{lag}'] = merged_pot_gdp['OutputGap'].shift(lag)

fedFundsTest = fedFundsTest[fedFundsTest['observation_date'] >= "2000-01-01"]
inflationTest = inflationTest[inflationTest['observation_date'] >= "2000-01-01"]
merged_pot_gdp = merged_pot_gdp[merged_pot_gdp['observation_date'] >= "2000-01-01"]

# Merge all datasets into one dataframe
merged_test_data = (
    fedFundsTest[['observation_date', 'FEDFUNDS', 'FEDFUNDS_Lag1', 'FEDFUNDS_Lag2', 'FEDFUNDS_Lag3']]
    .merge(inflationTest[['observation_date', 'Inflation_Rate', 'Inflation_Rate_Lag1', 'Inflation_Rate_Lag2', 'Inflation_Rate_Lag3']], on='observation_date')
    .merge(merged_pot_gdp[['observation_date', 'OutputGap', 'OutputGap_Lag1', 'OutputGap_Lag2', 'OutputGap_Lag3']], on='observation_date')
)


# Display merged data
print("Merged Test Data:")
print(merged_test_data.head())

# Standardize selected columns
standardizeCols = [
    "FEDFUNDS", "FEDFUNDS_Lag1", "FEDFUNDS_Lag2", "FEDFUNDS_Lag3",
    "Inflation_Rate", "Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
    "OutputGap", "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"
]

# Drop any NaNs before standardization
merged_test_data.dropna(subset=standardizeCols, inplace=True)

# Apply standardization
scaler = StandardScaler()
merged_test_data[standardizeCols] = scaler.fit_transform(merged_test_data[standardizeCols])

# Adjust the features to use 2000–2006 lagged variables
ols_models = [
    (ols_1997_without_lag_q1, "OLS 1997 Without Lagged FEDFUNDS (Q1)", ["Inflation_Rate", "OutputGap"]),
    (ols_1997_without_lag_all, "OLS 1997 Without Lagged FEDFUNDS (All Quarters)", ["Inflation_Rate", "OutputGap"]),
    (ols_1997_with_lag_q1, "OLS 1997 With Lagged FEDFUNDS (Q1)", ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"]),
    (ols_1997_with_lag_all, "OLS 1997 With Lagged FEDFUNDS (All Quarters)", ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"]),
    (ols_2002_without_lag_q1, "OLS 2002 Without Lagged FEDFUNDS (Q1)", ["Inflation_Rate", "OutputGap"]),
    (ols_2002_without_lag_all, "OLS 2002 Without Lagged FEDFUNDS (All Quarters)", ["Inflation_Rate", "OutputGap"]),
    (ols_2002_with_lag_q1, "OLS 2002 With Lagged FEDFUNDS (Q1)", ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"]),
    (ols_2002_with_lag_all, "OLS 2002 With Lagged FEDFUNDS (All Quarters)", ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"])
]

iv_models = [
    # IV model for 1997 without lagged FEDFUNDS (Q1)
    (results_1997_without_lagged_q1, "IV 1997 Without Lagged FEDFUNDS (Q1)", [],
     ["Inflation_Rate", "OutputGap"],
     ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
      "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),

    # IV model for 1997 without lagged FEDFUNDS (All Quarters)
    (results_1997_without_lagged_all, "IV 1997 Without Lagged FEDFUNDS (All Quarters)", [],
     ["Inflation_Rate", "OutputGap"],
     ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
      "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),

    # IV model for 1997 with lagged FEDFUNDS (Q1)
    (results_1997_with_lagged_q1, "IV 1997 With Lagged FEDFUNDS (Q1)", ["FEDFUNDS_Lag1"],
     ["Inflation_Rate", "OutputGap"],
     ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
      "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),

    # IV model for 1997 with lagged FEDFUNDS (All Quarters)
    (results_1997_with_lagged_all, "IV 1997 With Lagged FEDFUNDS (All Quarters)", ["FEDFUNDS_Lag1"],
     ["Inflation_Rate", "OutputGap"],
     ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
      "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),

    # IV model for 2002 without lagged FEDFUNDS (Q1)
    (results_2002_without_lagged_q1, "IV 2002 Without Lagged FEDFUNDS (Q1)", [],
     ["Inflation_Rate", "OutputGap"],
     ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
      "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),

    # IV model for 2002 without lagged FEDFUNDS (All Quarters)
    (results_2002_without_lagged_all, "IV 2002 Without Lagged FEDFUNDS (All Quarters)", [],
     ["Inflation_Rate", "OutputGap"],
     ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
      "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),

    # IV model for 2002 with lagged FEDFUNDS (Q1)
    (results_2002_with_lagged_q1, "IV 2002 With Lagged FEDFUNDS (Q1)", ["FEDFUNDS_Lag1"],
     ["Inflation_Rate", "OutputGap"],
     ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
      "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),

    # IV model for 2002 with lagged FEDFUNDS (All Quarters)
    (results_2002_with_lagged_all, "IV 2002 With Lagged FEDFUNDS (All Quarters)", ["FEDFUNDS_Lag1"],
     ["Inflation_Rate", "OutputGap"],
     ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
      "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"])
]
predictions = {}


def evaluate_models(test_data, ols_models, iv_models):
    """
    Evaluate OLS and IV models on the given test dataset.

    Args:
    - test_data (DataFrame): The test dataset with required columns.
    - ols_models (list): List of tuples with OLS models and metadata (name, features).
    - iv_models (list): List of tuples with IV models and metadata (name, exogenous, endogenous, instruments).

    Returns:
    - DataFrame: A DataFrame with evaluation results for all models.
    """
    evaluation_results = []

    # Evaluate OLS Models
    for model, model_name, features in ols_models:
        X_test = test_data[features]
        X_test = add_constant(X_test)  # Add constant if required
        y_test = test_data["FEDFUNDS"]

        # Predict and calculate metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - len(features) - 1))
        variance = y_test.var()
        mse_mean_ratio = mse / y_test.mean()

        evaluation_results.append({
            "Model": model_name,
            "Type": "OLS",
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "Adjusted R2": adj_r2,
            "Variance": variance,
            "MSE/Variance": mse / variance,
            "MSE/Mean": mse_mean_ratio
        })

        predictions[model_name] = y_pred

    # Evaluate IV Models
    for model, model_name, exog_features, endog_features, instruments in iv_models:
        print(f"Evaluating IV model: {model_name}")
        exog_test = add_constant(test_data[exog_features])  # Exogenous variables
        endog_test = test_data[endog_features]  # Endogenous variables
        y_test = test_data["FEDFUNDS"]
        print(exog_test.shape, endog_test.shape)
        print("#########################################################################")

        # Predict using linearmodels' IV predict method
        y_pred = model.predict(exog=exog_test, endog=endog_test)  # Include endog_test
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - len(exog_features) - 1))
        variance = y_test.var()
        mse_mean_ratio = mse / y_test.mean()

        evaluation_results.append({
            "Model": model_name,
            "Type": "IV",
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "Adjusted R2": adj_r2,
            "Variance": variance,
            "MSE/Variance": mse / variance,
            "MSE/Mean": mse_mean_ratio
        })

        predictions[model_name] = y_pred

    # Convert results to DataFrame
    return pd.DataFrame(evaluation_results), predictions


model_comparison_results = evaluate_models(merged_test_data, ols_models, iv_models)
print(model_comparison_results[0])

def save_df_as_image(df, filename="TestingResults2000-2006.png"):
    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5))  # Adjust size dynamically based on number of rows
    ax.axis("tight")
    ax.axis("off")

    # Create a table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    # Adjust font size and column widths
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save the image
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# Save the `model_comparison_results` DataFrame to an image
# save_df_as_image(model_comparison_results, "TestingResults2000-2006.png")

# print(merged_test_data.columns)
# print(merged_data[['FEDFUNDS_19970107', 'FEDFUNDS_19970107', 'OutputGap_1997']].describe())
# print(merged_test_data[['FEDFUNDS', 'Inflation_Rate', 'OutputGap']].describe())

plt.figure(figsize=(10, 6))
plt.plot(merged_test_data['observation_date'], merged_test_data['FEDFUNDS'], label="Actual FedFunds Values (1997)", alpha=0.6, color="pink")
plt.plot(merged_test_data['observation_date'], predictions["IV 1997 Without Lagged FEDFUNDS (All Quarters)"], label="Pred Vals Excl Lagged FedFunds (IV)", alpha=0.6, color="purple")
plt.plot(merged_test_data['observation_date'], predictions["OLS 1997 Without Lagged FEDFUNDS (All Quarters)"], label="Pred Vals Excl Lagged FedFunds (OLS)", alpha=0.6, color="red")
plt.plot(merged_test_data['observation_date'], predictions["IV 1997 With Lagged FEDFUNDS (All Quarters)"], label="Pred Vals Incl Lagged FedFunds (IV)", alpha=0.6, color="blue")
plt.plot(merged_test_data['observation_date'], predictions["OLS 1997 With Lagged FEDFUNDS (All Quarters)"], label="Pred Vals Incl Lagged FedFunds (OLS)", alpha=0.6, color="orange")
plt.title("Actual vs. Pred FedFunds Values (1997) Tested on 2000-2006")
plt.legend()
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(merged_test_data['observation_date'], merged_test_data['FEDFUNDS'], label="Actual FedFunds Values (2002)", alpha=0.6, color="pink")
plt.plot(merged_test_data['observation_date'], predictions["IV 2002 Without Lagged FEDFUNDS (All Quarters)"], label="Pred Vals Excl Lagged FedFunds (IV)", alpha=0.6, color="purple")
plt.plot(merged_test_data['observation_date'], predictions["OLS 2002 Without Lagged FEDFUNDS (All Quarters)"], label="Pred Vals Excl Lagged FedFunds (OLS)", alpha=0.6, color="red")
plt.plot(merged_test_data['observation_date'], predictions["IV 2002 With Lagged FEDFUNDS (All Quarters)"], label="Pred Vals Incl Lagged FedFunds (IV)", alpha=0.6, color="blue")
plt.plot(merged_test_data['observation_date'], predictions["OLS 2002 With Lagged FEDFUNDS (All Quarters)"], label="Pred Vals Incl Lagged FedFunds (OLS)", alpha=0.6, color="orange")
plt.title("Actual vs. Pred FedFunds Values (2002) Tested on 2000-2006")
plt.legend()
# plt.show()
