import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

from RegressionOilPrice import (
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
housingTest = pd.read_csv("HousingTest.csv")
oilTest = pd.read_csv("OilPriceTest.csv")
unemploymentTest = pd.read_csv("UnemploymentTest.csv")

fedFundsTest['observation_date'] = pd.to_datetime(fedFundsTest['observation_date'])
inflationTest['observation_date'] = pd.to_datetime(inflationTest['observation_date'])
realGDPTest['observation_date'] = pd.to_datetime(realGDPTest['observation_date'])
potGDPTest['observation_date'] = pd.to_datetime(potGDPTest['observation_date'])
housingTest['observation_date'] = pd.to_datetime(housingTest['observation_date'])
oilTest['observation_date'] = pd.to_datetime(oilTest['observation_date'])
unemploymentTest['observation_date'] = pd.to_datetime(unemploymentTest['observation_date'])

start_date = '1998-01-01'
end_date = '2006-12-31'

fedFundsTest = fedFundsTest[(fedFundsTest['observation_date'] >= start_date) & (fedFundsTest['observation_date'] <= end_date)]
inflationTest = inflationTest[(inflationTest['observation_date'] >= start_date) & (inflationTest['observation_date'] <= end_date)]
realGDPTest = realGDPTest[(realGDPTest['observation_date'] >= start_date) & (realGDPTest['observation_date'] <= end_date)]
potGDPTest = potGDPTest[(potGDPTest['observation_date'] >= start_date) & (potGDPTest['observation_date'] <= end_date)]
housingTest = housingTest[(housingTest['observation_date'] >= start_date) & (housingTest['observation_date'] <= end_date)]
oilTest = oilTest[(oilTest['observation_date'] >= start_date) & (oilTest['observation_date'] <= end_date)]
unemploymentTest = unemploymentTest[(unemploymentTest['observation_date'] >= start_date) & (unemploymentTest['observation_date'] <= end_date)]

# print("Filtered Fed Funds Data:")
# print(fedFundsTest.head())
# print("\nFiltered Inflation Data:")
# print(inflationTest.head())
# print("\nFiltered Real GDP Data:")
# print(realGDPTest.head())
# print("\nFiltered Potential GDP Data:")
# print(potGDPTest.head())

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
    # housingTest[f'Housing_Lag{lag}'] = housingTest['HOUST'].shift(lag)
    oilTest[f'Oil_Lag{lag}'] = oilTest['OILPRICE'].shift(lag)
    # unemploymentTest[f'Unemployment_Lag{lag}'] = unemploymentTest['UNRATE'].shift(lag)

fedFundsTest = fedFundsTest[fedFundsTest['observation_date'] >= "2000-01-01"]
inflationTest = inflationTest[inflationTest['observation_date'] >= "2000-01-01"]
merged_pot_gdp = merged_pot_gdp[merged_pot_gdp['observation_date'] >= "2000-01-01"]
housingTest = housingTest[housingTest['observation_date'] >= "2000-01-01"]
oilTest = oilTest[oilTest['observation_date'] >= "2000-01-01"]
unemploymentTest = unemploymentTest[unemploymentTest['observation_date'] >= "2000-01-01"]


# Merge all datasets into one dataframe
merged_test_data = (
    fedFundsTest[['observation_date', 'FEDFUNDS', 'FEDFUNDS_Lag1', 'FEDFUNDS_Lag2', 'FEDFUNDS_Lag3']]
    .merge(inflationTest[['observation_date', 'Inflation_Rate', 'Inflation_Rate_Lag1', 'Inflation_Rate_Lag2', 'Inflation_Rate_Lag3']], on='observation_date')
    .merge(merged_pot_gdp[['observation_date', 'OutputGap', 'OutputGap_Lag1', 'OutputGap_Lag2', 'OutputGap_Lag3']], on='observation_date')
    # .merge(housingTest[['observation_date', 'HOUST', 'Housing_Lag1', 'Housing_Lag2', 'Housing_Lag3']], on='observation_date')
    .merge(oilTest[['observation_date','OILPRICE', 'Oil_Lag1', 'Oil_Lag2', 'Oil_Lag3']], on='observation_date')
    # .merge(unemploymentTest[['observation_date', 'UNRATE', 'Unemployment_Lag1', 'Unemployment_Lag2', 'Unemployment_Lag3']], on='observation_date')
)


# print(inflationTest.head())
# print(fedFundsTest.head())
# print(merged_pot_gdp.head())
# print(housingTest.head())
# print(oilTest.head())
# print(unemploymentTest.head())
# # Display merged data
print("Merged Test Data:")
print(merged_test_data.head())


# # Adjust the features to use 2000–2006 lagged variables
# # ols_models = [
# #     (ols_1997_without_lag_q1, "OLS 1997 Without Lagged FEDFUNDS (Q1)", ["Inflation_Rate", "OutputGap"]),
# #     (ols_1997_without_lag_all, "OLS 1997 Without Lagged FEDFUNDS (All Quarters)", ["Inflation_Rate", "OutputGap"]),
# #     (ols_1997_with_lag_q1, "OLS 1997 With Lagged FEDFUNDS (Q1)", ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"]),
# #     (ols_1997_with_lag_all, "OLS 1997 With Lagged FEDFUNDS (All Quarters)", ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"]),
# #     (ols_2002_without_lag_q1, "OLS 2002 Without Lagged FEDFUNDS (Q1)", ["Inflation_Rate", "OutputGap"]),
# #     (ols_2002_without_lag_all, "OLS 2002 Without Lagged FEDFUNDS (All Quarters)", ["Inflation_Rate", "OutputGap"]),
# #     (ols_2002_with_lag_q1, "OLS 2002 With Lagged FEDFUNDS (Q1)", ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"]),
# #     (ols_2002_with_lag_all, "OLS 2002 With Lagged FEDFUNDS (All Quarters)", ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap"])
# # ]
ols_models = [
    # 1997 Without Lagged FEDFUNDS (Q1)
    (
        ols_1997_without_lag_q1,
        "OLS 1997 Without Lagged FEDFUNDS (Q1)",
        ["Inflation_Rate", "OutputGap", "OILPRICE"]
    ),
    # 1997 Without Lagged FEDFUNDS (All Quarters)
    (
        ols_1997_without_lag_all,
        "OLS 1997 Without Lagged FEDFUNDS (All Quarters)",
        ["Inflation_Rate", "OutputGap", "OILPRICE"]
    ),
    # 1997 With Lagged FEDFUNDS (Q1)
    (
        ols_1997_with_lag_q1,
        "OLS 1997 With Lagged FEDFUNDS (Q1)",
        ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap", "OILPRICE"]
    ),
    # 1997 With Lagged FEDFUNDS (All Quarters)
    (
        ols_1997_with_lag_all,
        "OLS 1997 With Lagged FEDFUNDS (All Quarters)",
        ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap", "OILPRICE"]
    ),
    # 2002 Without Lagged FEDFUNDS (Q1)
    (
        ols_2002_without_lag_q1,
        "OLS 2002 Without Lagged FEDFUNDS (Q1)",
        ["Inflation_Rate", "OutputGap", "OILPRICE"]
    ),
    # 2002 Without Lagged FEDFUNDS (All Quarters)
    (
        ols_2002_without_lag_all,
        "OLS 2002 Without Lagged FEDFUNDS (All Quarters)",
        ["Inflation_Rate", "OutputGap","OILPRICE"]
    ),
    # 2002 With Lagged FEDFUNDS (Q1)
    (
        ols_2002_with_lag_q1,
        "OLS 2002 With Lagged FEDFUNDS (Q1)",
        ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap", "OILPRICE"]
    ),
    # 2002 With Lagged FEDFUNDS (All Quarters)
    (
        ols_2002_with_lag_all,
        "OLS 2002 With Lagged FEDFUNDS (All Quarters)",
        ["FEDFUNDS_Lag1", "Inflation_Rate", "OutputGap", "OILPRICE"]
    )
]

# # iv_models = [
# #     # IV model for 1997 without lagged FEDFUNDS (Q1)
# #     (results_1997_without_lagged_q1, "IV 1997 Without Lagged FEDFUNDS (Q1)", [],
# #      ["Inflation_Rate", "OutputGap"],
# #      ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
# #       "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),
# #
# #     # IV model for 1997 without lagged FEDFUNDS (All Quarters)
# #     (results_1997_without_lagged_all, "IV 1997 Without Lagged FEDFUNDS (All Quarters)", [],
# #      ["Inflation_Rate", "OutputGap"],
# #      ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
# #       "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),
# #
# #     # IV model for 1997 with lagged FEDFUNDS (Q1)
# #     (results_1997_with_lagged_q1, "IV 1997 With Lagged FEDFUNDS (Q1)", ["FEDFUNDS_Lag1"],
# #      ["Inflation_Rate", "OutputGap"],
# #      ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
# #       "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),
# #
# #     # IV model for 1997 with lagged FEDFUNDS (All Quarters)
# #     (results_1997_with_lagged_all, "IV 1997 With Lagged FEDFUNDS (All Quarters)", ["FEDFUNDS_Lag1"],
# #      ["Inflation_Rate", "OutputGap"],
# #      ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
# #       "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),
# #
# #     # IV model for 2002 without lagged FEDFUNDS (Q1)
# #     (results_2002_without_lagged_q1, "IV 2002 Without Lagged FEDFUNDS (Q1)", [],
# #      ["Inflation_Rate", "OutputGap"],
# #      ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
# #       "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),
# #
# #     # IV model for 2002 without lagged FEDFUNDS (All Quarters)
# #     (results_2002_without_lagged_all, "IV 2002 Without Lagged FEDFUNDS (All Quarters)", [],
# #      ["Inflation_Rate", "OutputGap"],
# #      ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
# #       "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),
# #
# #     # IV model for 2002 with lagged FEDFUNDS (Q1)
# #     (results_2002_with_lagged_q1, "IV 2002 With Lagged FEDFUNDS (Q1)", ["FEDFUNDS_Lag1"],
# #      ["Inflation_Rate", "OutputGap"],
# #      ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
# #       "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),
# #
# #     # IV model for 2002 with lagged FEDFUNDS (All Quarters)
# #     (results_2002_with_lagged_all, "IV 2002 With Lagged FEDFUNDS (All Quarters)", ["FEDFUNDS_Lag1"],
# #      ["Inflation_Rate", "OutputGap"],
# #      ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
# #       "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"])
# # ]
iv_models = [
    # IV model for 1997 without lagged FEDFUNDS (Q1)
    (
        results_1997_without_lagged_q1,
        "IV 1997 Without Lagged FEDFUNDS (Q1)",
        [],
        ["Inflation_Rate", "OutputGap", "OILPRICE"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3",
         # "Houses_Lag1", "Houses_Lag2", "Houses_Lag3",
         "Oil_Price_Lag1", "Oil_Price_Lag2", "Oil_Price_Lag3"]
         # "Unemployment_Lag1", "Unemployment_Lag2", "Unemployment_Lag3"]
    ),

    # IV model for 1997 without lagged FEDFUNDS (All Quarters)
    (
        results_1997_without_lagged_all,
        "IV 1997 Without Lagged FEDFUNDS (All Quarters)",
        [],
        ["Inflation_Rate", "OutputGap", "OILPRICE"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3",
         # "Houses_Lag1", "Houses_Lag2", "Houses_Lag3",
         "Oil_Price_Lag1", "Oil_Price_Lag2", "Oil_Price_Lag3"]
         # "Unemployment_Lag1", "Unemployment_Lag2", "Unemployment_Lag3"]
    ),

    # IV model for 1997 with lagged FEDFUNDS (Q1)
    (
        results_1997_with_lagged_q1,
        "IV 1997 With Lagged FEDFUNDS (Q1)",
        ["FEDFUNDS_Lag1"],
        ["Inflation_Rate", "OutputGap", "OILPRICE"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3",
         # "Houses_Lag1",
         "Oil_Price_Lag1"]
         # "Unemployment_Lag1"]
    ),

    # IV model for 1997 with lagged FEDFUNDS (All Quarters)
    (
        results_1997_with_lagged_all,
        "IV 1997 With Lagged FEDFUNDS (All Quarters)",
        ["FEDFUNDS_Lag1"],
        ["Inflation_Rate", "OutputGap", "OILPRICE"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3",
         # "Houses_Lag1", "Houses_Lag2", "Houses_Lag3",
         "Oil_Price_Lag1", "Oil_Price_Lag2", "Oil_Price_Lag3"]
         # "Unemployment_Lag1", "Unemployment_Lag2", "Unemployment_Lag3"]
    ),

   # IV model for 2002 without lagged FEDFUNDS (Q1)
    (
        results_2002_without_lagged_q1,
        "IV 2002 Without Lagged FEDFUNDS (Q1)",
        [],
        ["Inflation_Rate", "OutputGap", "OILPRICE"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3",
         # "Houses_Lag1", "Houses_Lag2", "Houses_Lag3",
         "Oil_Price_Lag1", "Oil_Price_Lag2", "Oil_Price_Lag3"]
         # "Unemployment_Lag1", "Unemployment_Lag2", "Unemployment_Lag3"]
    ),


    # IV model for 2002 without lagged FEDFUNDS (All Quarters)
    (
        results_2002_without_lagged_all, "IV 2002 Without Lagged FEDFUNDS (All Quarters)", [],
        ["Inflation_Rate", "OutputGap", "OILPRICE"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3",
         # "Houses_Lag1", "Houses_Lag2", "Houses_Lag3",
         "Oil_Price_Lag1", "Oil_Price_Lag2", "Oil_Price_Lag3"]
         # "Unemployment_Lag1", "Unemployment_Lag2", "Unemployment_Lag3"]
    ),

# #     # IV model for 2002 without lagged FEDFUNDS (Q1)
# #     (results_2002_without_lagged_q1, "IV 2002 Without Lagged FEDFUNDS (Q1)", [],
# #      ["Inflation_Rate", "OutputGap"],
# #      ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
# #       "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),
# #
# #     # IV model for 2002 without lagged FEDFUNDS (All Quarters)
# #     (results_2002_without_lagged_all, "IV 2002 Without Lagged FEDFUNDS (All Quarters)", [],
# #      ["Inflation_Rate", "OutputGap"],
# #      ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
# #       "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3"]),
    # IV model for 2002 with lagged FEDFUNDS (Q1)
    (
        results_2002_with_lagged_q1,
        "IV 2002 With Lagged FEDFUNDS (Q1)",
        ["FEDFUNDS_Lag1"],
        ["Inflation_Rate", "OutputGap", "OILPRICE"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3",
         # "Houses_Lag1",
         "Oil_Price_Lag1"]
         # "Unemployment_Lag1"]
    ),

    # IV model for 2002 with lagged FEDFUNDS (All Quarters)
    (
        results_2002_with_lagged_all,
        "IV 2002 With Lagged FEDFUNDS (All Quarters)",
        ["FEDFUNDS_Lag1"],
        ["Inflation_Rate", "OutputGap","OILPRICE"],
        ["Inflation_Rate_Lag1", "Inflation_Rate_Lag2", "Inflation_Rate_Lag3",
         "OutputGap_Lag1", "OutputGap_Lag2", "OutputGap_Lag3",
         # "Houses_Lag1", "Houses_Lag2", "Houses_Lag3",
         "Oil_Price_Lag1", "Oil_Price_Lag2", "Oil_Price_Lag3"]
         # "Unemployment_Lag1", "Unemployment_Lag2", "Unemployment_Lag3"]
    )
]

# for i, model_tuple in enumerate(iv_models):
#     print(f"Model {i}: {len(model_tuple)} elements")

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

    # Convert results to DataFrame
    return pd.DataFrame(evaluation_results)


model_comparison_results = evaluate_models(merged_test_data, ols_models, iv_models)
print(model_comparison_results)
#
# def save_df_as_image(df, filename):
#     # Create a Matplotlib figure
#     fig, ax = plt.subplots(figsize=(12, len(df) * 0.5))  # Adjust size dynamically based on number of rows
#     ax.axis("tight")
#     ax.axis("off")
#
#     # Create a table
#     table = ax.table(
#         cellText=df.values,
#         colLabels=df.columns,
#         loc="center",
#         cellLoc="center"
#     )
#
#     # Adjust font size and column widths
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.auto_set_column_width(col=list(range(len(df.columns))))
#
#     # Save the image
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# # Save the `model_comparison_results` DataFrame to an image
# save_df_as_image(model_comparison_results, "TestingResults2000-2006.png")
#
# print(merged_test_data.columns)

# print(merged_data[['FEDFUNDS_19970107', 'FEDFUNDS_19970107', 'OutputGap_1997']].describe())
# print(merged_test_data[['FEDFUNDS', 'Inflation_Rate', 'OutputGap']].describe())
