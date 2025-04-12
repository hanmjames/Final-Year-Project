import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set pandas display option to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# Load the separate datasets for real GDP, potential GDP, inflation, and federal funds rate
real_gdp = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\RealGDP_1981to1996.csv")
pot_gdp = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\PotGDP_1981to1996.csv")
inflation = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\Inflation_1981to1996.csv")
fed_funds = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\Fedfunds_1981to1996.csv")
oil_prices = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\OilPrice_Quarterly.csv")
unemployment_rates = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\UnemploymentRate_Quarterly.csv")
housing = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\Housing_Quarterly.csv")

oil_prices = oil_prices[pd.to_datetime(oil_prices['observation_date']) >= '1970-01-01']
unemployment_rates = unemployment_rates[pd.to_datetime(unemployment_rates['observation_date']) >= '1970-01-01']
housing = housing[pd.to_datetime(housing['observation_date']) >= '1970-01-01']


# Merge datasets on 'observation_date'
merged_data = (
    real_gdp.merge(pot_gdp, on="observation_date")
    .merge(inflation, on="observation_date")
    .merge(fed_funds, on="observation_date")
    # .merge(oil_prices, on="observation_date")
    .merge(housing, on="observation_date")
    # .merge(unemployment_rates, on="observation_date")
)

# Convert 'observation_date' to datetime format for consistency
merged_data['observation_date'] = pd.to_datetime(merged_data['observation_date'])

# Calculate the annualized inflation rates for 1997 and 2002 before filtering the data
merged_data['Inflation_Rate_1997'] = (
    (merged_data['GDPDEF_19970131'] / merged_data['GDPDEF_19970131'].shift(4) - 1) * 100
)
merged_data['Inflation_Rate_2002'] = (
    (merged_data['GDPDEF_20020130'] / merged_data['GDPDEF_20020130'].shift(4) - 1) * 100
)

# Calculate the output gap for both vintages before filtering
merged_data['OutputGap_1997'] = 100 * (
    np.log(merged_data['GDPC1_19970131']) - np.log(merged_data['GDPPOT_19970128'])
)
merged_data['OutputGap_2002'] = 100 * (
    np.log(merged_data['GDPC1_20020130']) - np.log(merged_data['GDPPOT_20020201'])
)

# Generate explicit lag columns for IV regression
for lag in range(1, 4):
    merged_data[f'OutputGap_1997_Lag{lag}'] = merged_data['OutputGap_1997'].shift(lag)
    merged_data[f'Inflation_Rate_1997_Lag{lag}'] = merged_data['Inflation_Rate_1997'].shift(lag)
    merged_data[f'FedFunds_1997_Lag{lag}'] = merged_data['FEDFUNDS_19970107'].shift(lag)
    # merged_data[f'Oil_Price_1997_Lag{lag}'] = merged_data['OILPRICE_19970303'].shift(lag)
    merged_data[f'Houses_1997_Lag{lag}'] = merged_data['HOUST_19970122'].shift(lag)
    # merged_data[f'Unemployment_1997_Lag{lag}'] = merged_data['UNRATE_19970110'].shift(lag)
    merged_data[f'OutputGap_2002_Lag{lag}'] = merged_data['OutputGap_2002'].shift(lag)
    merged_data[f'Inflation_Rate_2002_Lag{lag}'] = merged_data['Inflation_Rate_2002'].shift(lag)
    merged_data[f'FedFunds_2002_Lag{lag}'] = merged_data['FEDFUNDS_20020108'].shift(lag)
    # merged_data[f'Oil_Price_2002_Lag{lag}'] = merged_data['OILPRICE_20020102'].shift(lag)
    merged_data[f'Houses_2002_Lag{lag}'] = merged_data['HOUST_20020117'].shift(lag)
    # merged_data[f'Unemployment_2002_Lag{lag}'] = merged_data['UNRATE_20020104'].shift(lag)

# Restrict the range of analysis to 1981-1996 (include all quarters)
merged_data = merged_data[(merged_data['observation_date'] >= '1981-01-01') & (merged_data['observation_date'] <= '1996-12-31')]

# Ensure 'observation_date' is retained in the merged data
merged_data.reset_index(drop=True, inplace=True)

# Display the updated merged data with output gaps and inflation rate
print("Merged Data (After Adding Unemployment, Housing, And Oil Prices):")
print(merged_data.head())

# Filter the data to include only the first day of January for each year (16 points for analysis)
merged_data_q1 = merged_data.loc[merged_data['observation_date'].dt.strftime('%m-%d') == '01-01'].copy()

# Display merged_data_q1 to confirm the selection of Q1 only
print("Filtered Data for Q1 (First Day of Each Year):")
print(merged_data_q1)

# Define columns to standardize
standardize_cols = [
    # 1997
    "FEDFUNDS_19970107", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122",
    "FedFunds_1997_Lag1",
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3",

    # 2002
    "FEDFUNDS_20020108", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117",
    "FedFunds_2002_Lag1",
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
    "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3"
]

# Drop rows with NaNs before standardization (safe practice before model fitting too)
merged_data.dropna(subset=standardize_cols, inplace=True)
merged_data_q1 = merged_data[merged_data['observation_date'].dt.strftime('%m-%d') == '01-01'].copy()

# Standardize using sklearn
scaler = StandardScaler()
merged_data[standardize_cols] = scaler.fit_transform(merged_data[standardize_cols])
merged_data_q1[standardize_cols] = scaler.transform(merged_data_q1[standardize_cols])


# Function to save regression summary as image
# def save_summary(summary, title, file_name):
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.text(0, 0, str(summary), fontsize=10, family="monospace")
#     ax.axis("off")
#     ax.set_title(title, fontsize=12)
#     plt.savefig(file_name, bbox_inches="tight")
def save_summary(summary, title, file_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0, 0, str(summary), fontsize=10, family="monospace")
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    plt.savefig(file_name, bbox_inches="tight")
# OLS Regressions
# OLS Regression for 1997 Without Lagged FedFunds (First Quarter Only) -1
y_1997_q1 = merged_data_q1["FEDFUNDS_19970107"]
X_1997_q1 = merged_data_q1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]]
X_1997_q1 = add_constant(X_1997_q1)

ols_1997_without_lag_q1 = OLS(y_1997_q1, X_1997_q1).fit()
print("OLS Results for 1997 (Without Lagged FedFunds, First Quarter Only):")
print(ols_1997_without_lag_q1.summary())

# OLS Regression for 1997 Without Lagged FedFunds (All Quarters) -2
y_1997_all = merged_data["FEDFUNDS_19970107"]
X_1997_all = merged_data[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]]
X_1997_all = add_constant(X_1997_all)

ols_1997_without_lag_all = OLS(y_1997_q1, X_1997_q1).fit()
print("OLS Results for 1997 (Without Lagged FedFunds, All Quarters):")
print(ols_1997_without_lag_all.summary())

# OLS Regression for 1997 With Lagged FedFunds (First Quarter Only) -5
y_1997_q1_lagged = merged_data_q1["FEDFUNDS_19970107"]
X_1997_q1_lagged = add_constant(merged_data_q1[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]])

ols_1997_with_lag_q1 = OLS(y_1997_q1_lagged, X_1997_q1_lagged).fit()
print("OLS Results for 1997 (With Lagged FedFunds, First Quarter Only):")
print(ols_1997_with_lag_q1.summary())

# OLS Regression for 1997 With Lagged FedFunds (All Quarters) -6
y_1997_all_lagged = merged_data["FEDFUNDS_19970107"]
X_1997_all_lagged = add_constant(merged_data[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]])

ols_1997_with_lag_all = OLS(y_1997_all_lagged, X_1997_all_lagged).fit()
print("OLS Results for 1997 (With Lagged FedFunds, All Quarters):")
print(ols_1997_with_lag_all.summary())

# OLS Regression for 2002 Without Lagged FedFunds (First Quarter Only) -7
y_2002_q1 = merged_data_q1["FEDFUNDS_20020108"]
X_2002_q1 = merged_data_q1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]]
X_2002_q1 = add_constant(X_2002_q1)

ols_2002_without_lag_q1 = OLS(y_2002_q1, X_2002_q1).fit()
print("OLS Results for 2002 (Without Lagged FedFunds, First Quarter Only):")
print(ols_2002_without_lag_q1.summary())

# OLS Regression for 2002 Without Lagged FedFunds (All Quarters) -8
y_2002_all = merged_data["FEDFUNDS_20020108"]
X_2002_all = merged_data[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]]
X_2002_all = add_constant(X_2002_all)

ols_2002_without_lag_all = OLS(y_2002_all, X_2002_all).fit()
print("OLS Results for 2002 (Without Lagged FedFunds, All Quarters):")
print(ols_2002_without_lag_all.summary())

# OLS Regression for 2002 With Lagged FedFunds (First Quarter Only) -9
y_2002_q1_lagged = merged_data_q1["FEDFUNDS_20020108"]
X_2002_q1_lagged = add_constant(merged_data_q1[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]])

ols_2002_with_lag_q1 = OLS(y_2002_q1_lagged, X_2002_q1_lagged).fit()
print("OLS Results for 2002 (With Lagged FedFunds, First Quarter Only):")
print(ols_2002_with_lag_q1.summary())

# OLS Regression for 2002 With Lagged FedFunds (All Quarters) -10
y_2002_all_lagged = merged_data["FEDFUNDS_20020108"]
X_2002_all_lagged = add_constant(merged_data[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]])

ols_2002_with_lag_all = OLS(y_2002_all_lagged, X_2002_all_lagged).fit()
print("OLS Results for 2002 (With Lagged FedFunds, All Quarters):")
print(ols_2002_with_lag_all.summary())



# IV Regressions
# IV Regression for 1997 Without Lagged FedFunds (First Quarter Only)
dependent_1997_q1 = merged_data_q1["FEDFUNDS_19970107"]
exog_1997_q1 = add_constant(merged_data_q1[[]])  # Only the constant term remains
endog_1997_q1 = merged_data_q1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]]
instruments_1997_q1 = merged_data_q1[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3"
]]
iv_model_1997_q1 = IV2SLS(dependent_1997_q1, exog_1997_q1, endog_1997_q1, instruments_1997_q1)
results_1997_without_lagged_q1 = iv_model_1997_q1.fit()
print("IV Regression Results for 1997 Vintage Excl. Lagged FEDFUNDS (First Quarter Only):")
print(results_1997_without_lagged_q1.summary)


# IV Regression for 1997 Without Lagged FedFunds (All Quarters)
dependent_1997_all = merged_data["FEDFUNDS_19970107"]
exog_1997_all = add_constant(merged_data[[]])  # Only the constant term remains
endog_1997_all = merged_data[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]]
instruments_1997_all = merged_data[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3"
]]
iv_model_1997_all = IV2SLS(dependent_1997_all, exog_1997_all, endog_1997_all, instruments_1997_all)
results_1997_without_lagged_all = iv_model_1997_all.fit()
print("IV Regression Results for 1997 Vintage Excl. Lagged FEDFUNDS (All Quarters):")
print(results_1997_without_lagged_all.summary)

# IV Regression for 1997 With Lagged FedFunds (First Quarter Only)
dependent_1997_q1_lagged = merged_data_q1["FEDFUNDS_19970107"]
exog_1997_q1_lagged = add_constant(merged_data_q1[["FedFunds_1997_Lag1"]])
endog_1997_q1_lagged = merged_data_q1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]]

instruments_1997_q1_lagged = merged_data_q1[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Houses_1997_Lag1"
]]
iv_model_1997_q1_lagged = IV2SLS(dependent_1997_q1_lagged, exog_1997_q1_lagged, endog_1997_q1_lagged, instruments_1997_q1_lagged)
results_1997_with_lagged_q1 = iv_model_1997_q1_lagged.fit()
print("IV Regression Results for 1997 Vintage Incl. Lagged FEDFUNDS (First Quarter Only):")
print(results_1997_with_lagged_q1.summary)

# IV Regression for 1997 With Lagged FedFunds (All Quarters)
dependent_1997_all_lagged = merged_data["FEDFUNDS_19970107"]
exog_1997_all_lagged = add_constant(merged_data[["FedFunds_1997_Lag1"]])
endog_1997_all_lagged = merged_data[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]]
instruments_1997_all_lagged = merged_data[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3",
    "Houses_1997_Lag1", "Houses_1997_Lag2", "Houses_1997_Lag3"
]]
iv_model_1997_all_lagged = IV2SLS(dependent_1997_all_lagged, exog_1997_all_lagged, endog_1997_all_lagged, instruments_1997_all_lagged)
results_1997_with_lagged_all = iv_model_1997_all_lagged.fit()
print("IV Regression Results for 1997 Vintage Incl. Lagged FEDFUNDS (All Quarters):")
print(results_1997_with_lagged_all.summary)


# IV Regression for 2002 Without Lagged FedFunds (First Quarter Only)
dependent_2002_q1 = merged_data_q1["FEDFUNDS_20020108"]
exog_2002_q1 = add_constant(merged_data_q1[[]])  # Only the constant term remains
endog_2002_q1 = merged_data_q1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]]
instruments_2002_q1 = merged_data_q1[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
    "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3"
]]
iv_model_2002_q1 = IV2SLS(dependent_2002_q1, exog_2002_q1, endog_2002_q1, instruments_2002_q1)
results_2002_without_lagged_q1 = iv_model_2002_q1.fit()
print("IV Regression Results for 2002 Vintage Excl. Lagged FEDFUNDS (First Quarter Only):")
print(results_2002_without_lagged_q1.summary)

# IV Regression for 2002 Without Lagged FedFunds (All Quarters)
dependent_2002_all = merged_data["FEDFUNDS_20020108"]
exog_2002_all = add_constant(merged_data[[]])  # Only the constant term remains
endog_2002_all = merged_data[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]]
instruments_2002_all = merged_data[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
    "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3"
]]
iv_model_2002_all = IV2SLS(dependent_2002_all, exog_2002_all, endog_2002_all, instruments_2002_all)
results_2002_without_lagged_all = iv_model_2002_all.fit()
print("IV Regression Results for 2002 Vintage Excl. Lagged FEDFUNDS (All Quarters):")
print(results_2002_without_lagged_all.summary)


# IV Regression for 2002 With Lagged FedFunds (First Quarter Only)
dependent_2002_q1_lagged = merged_data_q1["FEDFUNDS_20020108"]
exog_2002_q1_lagged = add_constant(merged_data_q1[["FedFunds_2002_Lag1"]])
endog_2002_q1_lagged = merged_data_q1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]]

instruments_2002_q1_lagged = merged_data_q1[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
    "Houses_2002_Lag1"
]]
iv_model_2002_q1_lagged = IV2SLS(dependent_2002_q1_lagged, exog_2002_q1_lagged, endog_2002_q1_lagged, instruments_2002_q1_lagged)
results_2002_with_lagged_q1 = iv_model_2002_q1_lagged.fit()
print("IV Regression Results for 2002 Vintage Incl. Lagged FEDFUNDS (First Quarter Only):")
print(results_2002_with_lagged_q1.summary)

# IV Regression for 2002 With Lagged FedFunds (All Quarters)
dependent_2002_all_lagged = merged_data["FEDFUNDS_20020108"]
exog_2002_all_lagged = add_constant(merged_data[["FedFunds_2002_Lag1"]])
endog_2002_all_lagged = merged_data[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]]
instruments_2002_all_lagged = merged_data[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3",
    "Houses_2002_Lag1", "Houses_2002_Lag2", "Houses_2002_Lag3"
]]
iv_model_2002_all_lagged = IV2SLS(dependent_2002_all_lagged, exog_2002_all_lagged, endog_2002_all_lagged, instruments_2002_all_lagged)
results_2002_with_lagged_all = iv_model_2002_all_lagged.fit()
print("IV Regression Results for 2002 Vintage Incl. Lagged FEDFUNDS (All Quarters):")
print(results_2002_with_lagged_all.summary)


#Save summaries as images
# save_summary(results_1997_without_lagged_q1.summary, "Adding 3 New Vars: IV Regression for 1997 Without Lagged FedFunds (First Quarter Only)", "iv_1997_without_lag_q1_3vars.png")
# save_summary(results_1997_without_lagged_all.summary, "Adding 3 New Vars: IV Regression for 1997 Without Lagged FedFunds (All Quarters)", "iv_1997_without_lag_all_3vars.png")
# save_summary(results_1997_with_lagged_q1.summary, "Adding 3 New Vars: IV Regression for 1997 With Lagged FedFunds (First Quarter Only)", "iv_1997_with_lag_q1_3vars.png")
# save_summary(results_1997_with_lagged_all.summary, "Adding 3 New Vars: IV Regression for 1997 With Lagged FedFunds (All Quarters)", "iv_1997_with_lag_all_3vars.png")
# save_summary(results_2002_without_lagged_q1.summary, "Adding 3 New Vars: IV Regression for 2002 Without Lagged FedFunds (First Quarter Only)", "iv_2002_without_lag_q1_3vars.png")
# save_summary(results_2002_without_lagged_all.summary, "Adding 3 New Vars: IV Regression for 2002 Without Lagged FedFunds (All Quarters)", "iv_2002_without_lag_all_3vars.png")
# save_summary(results_2002_with_lagged_q1.summary, "Adding 3 New Vars: IV Regression for 2002 With Lagged FedFunds (First Quarter Only)", "iv_2002_with_lag_q1_3vars.png")
# save_summary(results_2002_with_lagged_all.summary, "Adding 3 New Vars: IV Regression for 2002 With Lagged FedFunds (All Quarters)", "iv_2002_with_lag_all_3vars.png")
#
# # Extracting first-stage results for 1997 Without Lagged FedFunds (First Quarter Only)
# print("First-Stage Results for 1997 Without Lagged FedFunds (First Quarter Only):")
# print(results_1997_without_lagged_q1.first_stage)
#
# # Extracting first-stage results for 1997 Without Lagged FedFunds (All Quarters)
# print("First-Stage Results for 1997 Without Lagged FedFunds (All Quarters):")
# print(results_1997_without_lagged_all.first_stage)
#
# # Extracting first-stage results for 1997 With Lagged FedFunds (First Quarter Only)
# print("First-Stage Results for 1997 With Lagged FedFunds (First Quarter Only):")
# print(results_1997_with_lagged_q1.first_stage)
#
# # Extracting first-stage results for 1997 With Lagged FedFunds (All Quarters)
# print("First-Stage Results for 1997 With Lagged FedFunds (All Quarters):")
# print(results_1997_with_lagged_all.first_stage)
#
# # Extracting first-stage results for 2002 Without Lagged FedFunds (First Quarter Only)
# print("First-Stage Results for 2002 Without Lagged FedFunds (First Quarter Only):")
# print(results_2002_without_lagged_q1.first_stage)
#
# # Extracting first-stage results for 2002 Without Lagged FedFunds (All Quarters)
# print("First-Stage Results for 2002 Without Lagged FedFunds (All Quarters):")
# print(results_2002_without_lagged_all.first_stage)
#
# # Extracting first-stage results for 2002 With Lagged FedFunds (First Quarter Only)
# print("First-Stage Results for 2002 With Lagged FedFunds (First Quarter Only):")
# print(results_2002_with_lagged_q1.first_stage)
#
# # Extracting first-stage results for 2002 With Lagged FedFunds (All Quarters)
# print("First-Stage Results for 2002 With Lagged FedFunds (All Quarters):")
# print(results_2002_with_lagged_all.first_stage)
#
# save_summary(results_1997_without_lagged_q1.first_stage, "First-Stage Results for 1997 Without Lagged FedFunds (First Quarter Only)", "first_stage_1997_without_lag_q1_3vars.png")
#
# save_summary(results_1997_without_lagged_all.first_stage, "First-Stage Results for 1997 Without Lagged FedFunds (All Quarters)", "first_stage_1997_without_lag_all_3vars.png")
#
# save_summary(results_1997_with_lagged_q1.first_stage, "First-Stage Results for 1997 With Lagged FedFunds (First Quarter Only)", "first_stage_1997_with_lag_q1_3vars.png")
#
# save_summary(results_1997_with_lagged_all.first_stage, "First-Stage Results for 1997 With Lagged FedFunds (All Quarters)", "first_stage_1997_with_lag_all_3vars.png")
#
# save_summary(results_2002_without_lagged_q1.first_stage, "First-Stage Results for 2002 Without Lagged FedFunds (First Quarter Only)", "first_stage_2002_without_lag_q1_3vars.png")
#
# save_summary(results_2002_without_lagged_all.first_stage, "First-Stage Results for 2002 Without Lagged FedFunds (All Quarters)", "first_stage_2002_without_lag_all_3vars.png")
#
# save_summary(results_2002_with_lagged_q1.first_stage, "First-Stage Results for 2002 With Lagged FedFunds (First Quarter Only)", "first_stage_2002_with_lag_q1_3vars.png")
#
# save_summary(results_2002_with_lagged_all.first_stage, "First-Stage Results for 2002 With Lagged FedFunds (All Quarters)", "first_stage_2002_with_lag_all_3vars.png")

# Define the IV regression results
results = [
    {"Vintage": "1997", "Regression Type": "IV with Lagged FEDFUNDS (1997)", "Model": results_1997_with_lagged_q1},
    {"Vintage": "1997", "Regression Type": "IV without Lagged FEDFUNDS (1997)", "Model": results_1997_without_lagged_q1},
    {"Vintage": "1997", "Regression Type": "IV with Lagged FEDFUNDS (1997, All Quarters)", "Model": results_1997_with_lagged_all},
    {"Vintage": "1997", "Regression Type": "IV without Lagged FEDFUNDS (1997, All Quarters)", "Model": results_1997_without_lagged_all},
    {"Vintage": "2002", "Regression Type": "IV with Lagged FEDFUNDS (2002)", "Model": results_2002_with_lagged_q1},
    {"Vintage": "2002", "Regression Type": "IV without Lagged FEDFUNDS (2002)", "Model": results_2002_without_lagged_q1},
    {"Vintage": "2002", "Regression Type": "IV with Lagged FEDFUNDS (2002, All Quarters)", "Model": results_2002_with_lagged_all},
    {"Vintage": "2002", "Regression Type": "IV without Lagged FEDFUNDS (2002, All Quarters)", "Model": results_2002_without_lagged_all},
]

# Extract relevant statistics
table_data = []
for res in results:
    model = res["Model"]
    table_data.append({
        "Vintage": res["Vintage"],
        "Regression Type": res["Regression Type"],
        "No. of Observations": model.nobs,
        "R-squared": model.rsquared,
        "F-statistic": model.f_statistic.stat,
        "P-value (F-stat)": model.f_statistic.pval,
        "Sargan Test Statistic": model.sargan.stat if model.sargan is not None else "N/A",
        "Sargan P-value": model.sargan.pval if model.sargan is not None else "N/A"
    })

# Create a DataFrame
iv_comparison_df = pd.DataFrame(table_data)
print(iv_comparison_df)

import pandas as pd
from statsmodels.api import OLS, add_constant
import statsmodels.api as sm

# Creating a DataFrame to store OLS summary results
ols_results_data = []

# Adding each OLS model and extracting relevant statistics
models = [
    (ols_1997_without_lag_q1, "1997", "OLS Without Lagged FEDFUNDS (First Quarter Only)"),
    (ols_1997_without_lag_all, "1997", "OLS Without Lagged FEDFUNDS (All Quarters)"),
    (ols_1997_with_lag_q1, "1997", "OLS With Lagged FEDFUNDS (First Quarter Only)"),
    (ols_1997_with_lag_all, "1997", "OLS With Lagged FEDFUNDS (All Quarters)"),
    (ols_2002_without_lag_q1, "2002", "OLS Without Lagged FEDFUNDS (First Quarter Only)"),
    (ols_2002_without_lag_all, "2002", "OLS Without Lagged FEDFUNDS (All Quarters)"),
    (ols_2002_with_lag_q1, "2002", "OLS With Lagged FEDFUNDS (First Quarter Only)"),
    (ols_2002_with_lag_all, "2002", "OLS With Lagged FEDFUNDS (All Quarters)")
]

for model, vintage, regression_type in models:
    ols_results_data.append({
        "Vintage": vintage,
        "Regression Type": regression_type,
        "No. of Observations": model.nobs,
        "R-squared": model.rsquared,
        "Adjusted R-squared": model.rsquared_adj,
        "F-statistic": model.fvalue,
        "P-value (F-stat)": model.f_pvalue,
        "AIC": model.aic,
        "BIC": model.bic,
        "Durbin-Watson": sm.stats.durbin_watson(model.resid)  # Add Durbin-Watson statistic
    })

# Creating the DataFrame
ols_results_df = pd.DataFrame(ols_results_data)

# Displaying the DataFrame
print(ols_results_df)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from statsmodels.api import add_constant

# Splitting merged_data_q1 into training and testing sets for Q1 models
train_data_q1, test_data_q1 = train_test_split(merged_data_q1, test_size=0.2, random_state=42)

# Splitting merged_data into training and testing sets for all-quarter models
train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)

def calculate_metrics_iv(model, X_train_exog, X_train_endog, y_train, X_test_exog, X_test_endog, y_test, model_name):
    X_train_exog = add_constant(X_train_exog) if "const" in model.params.index else X_train_exog
    X_test_exog = add_constant(X_test_exog) if "const" in model.params.index else X_test_exog

    # In-sample predictions
    y_train_pred = model.predict(exog=X_train_exog, endog=X_train_endog)
    # Out-of-sample predictions
    y_test_pred = model.predict(exog=X_test_exog, endog=X_test_endog)

    # Metrics
    metrics = {
        "Model": model_name,
        "In-sample R-squared": model.rsquared,
        "In-sample RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "In-sample MAE": mean_absolute_error(y_train, y_train_pred),
        "Out-of-sample RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Out-of-sample MAE": mean_absolute_error(y_test, y_test_pred)
    }
    return metrics


# Function to calculate metrics for in-sample and out-of-sample testing
def calculate_metrics(model, X_train, y_train, X_test, y_test, model_name):
    X_train = add_constant(X_train) if "const" in model.params.index else X_train
    X_test = add_constant(X_test) if "const" in model.params.index else X_test

    # In-sample predictions
    y_train_pred = model.predict(X_train)
    # Out-of-sample predictions
    y_test_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "Model": model_name,
        "In-sample R-squared": model.rsquared,
        "In-sample RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "In-sample MAE": mean_absolute_error(y_train, y_train_pred),
        "Out-of-sample RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Out-of-sample MAE": mean_absolute_error(y_test, y_test_pred)
    }
    return metrics


# Updated OLS Models
ols_models = [
    # OLS 1997 Without Lagged FEDFUNDS (First Quarter Only)
    (ols_1997_without_lag_q1,
     train_data_q1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     train_data_q1["FEDFUNDS_19970107"],
     test_data_q1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     test_data_q1["FEDFUNDS_19970107"],
     "OLS 1997 Without Lagged FEDFUNDS (Q1)"),

    # OLS 1997 Without Lagged FEDFUNDS (All Quarters)
    (ols_1997_without_lag_all,
     train_data[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     train_data["FEDFUNDS_19970107"],
     test_data[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     test_data["FEDFUNDS_19970107"],
     "OLS 1997 Without Lagged FEDFUNDS (All Quarters)"),

    # OLS 1997 With Lagged FEDFUNDS (First Quarter Only)
    (ols_1997_with_lag_q1,
     train_data_q1[
         ["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     train_data_q1["FEDFUNDS_19970107"],
     test_data_q1[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     test_data_q1["FEDFUNDS_19970107"],
     "OLS 1997 With Lagged FEDFUNDS (Q1)"),

    # OLS 1997 With Lagged FEDFUNDS (All Quarters)
    (ols_1997_with_lag_all,
     train_data[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     train_data["FEDFUNDS_19970107"],
     test_data[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     test_data["FEDFUNDS_19970107"],
     "OLS 1997 With Lagged FEDFUNDS (All Quarters)"),

    # OLS 2002 Without Lagged FEDFUNDS (First Quarter Only)
    (ols_2002_without_lag_q1,
     train_data_q1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     train_data_q1["FEDFUNDS_20020108"],
     test_data_q1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     test_data_q1["FEDFUNDS_20020108"],
     "OLS 2002 Without Lagged FEDFUNDS (Q1)"),

    # OLS 2002 Without Lagged FEDFUNDS (All Quarters)
    (ols_2002_without_lag_all,
     train_data[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     train_data["FEDFUNDS_20020108"],
     test_data[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     test_data["FEDFUNDS_20020108"],
     "OLS 2002 Without Lagged FEDFUNDS (All Quarters)"),

    # OLS 2002 With Lagged FEDFUNDS (First Quarter Only)
    (ols_2002_with_lag_q1,
     train_data_q1[
         ["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     train_data_q1["FEDFUNDS_20020108"],
     test_data_q1[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     test_data_q1["FEDFUNDS_20020108"],
     "OLS 2002 With Lagged FEDFUNDS (Q1)"),

    # OLS 2002 With Lagged FEDFUNDS (All Quarters)
    (ols_2002_with_lag_all,
     train_data[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     train_data["FEDFUNDS_20020108"],
     test_data[["FedFunds_2002_Lag1", "Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     test_data["FEDFUNDS_20020108"],
     "OLS 2002 With Lagged FEDFUNDS (All Quarters)")
]

iv_models = [
    # IV 1997 Without Lagged FEDFUNDS (First Quarter Only)
    (results_1997_without_lagged_q1,
     train_data_q1[[]],  # Exogenous variables for training
     train_data_q1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     # Endogenous variables for training
     train_data_q1["FEDFUNDS_19970107"],  # Dependent variable for training
     test_data_q1[[]],  # Exogenous variables for testing
     test_data_q1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     # Endogenous variables for testing
     test_data_q1["FEDFUNDS_19970107"],  # Dependent variable for testing
     "IV 1997 Without Lagged FEDFUNDS (Q1)"
     ),

    # IV 1997 Without Lagged FEDFUNDS (All Quarters)
    (results_1997_without_lagged_all,
     train_data[[]],  # Exogenous variables for training
     train_data[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     # Endogenous variables for training
     train_data["FEDFUNDS_19970107"],  # Dependent variable for training
     test_data[[]],  # Exogenous variables for testing
     test_data[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     # Endogenous variables for testing
     test_data["FEDFUNDS_19970107"],  # Dependent variable for testing
     "IV 1997 Without Lagged FEDFUNDS (All Quarters)"
     ),

    # IV 1997 With Lagged FEDFUNDS (First Quarter Only)
    (results_1997_with_lagged_q1,
     train_data_q1[["FedFunds_1997_Lag1"]],  # Exogenous variables for training
     train_data_q1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     # Endogenous variables for training
     train_data_q1["FEDFUNDS_19970107"],  # Dependent variable for training
     test_data_q1[["FedFunds_1997_Lag1"]],  # Exogenous variables for testing
     test_data_q1[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     # Endogenous variables for testing
     test_data_q1["FEDFUNDS_19970107"],  # Dependent variable for testing
     "IV 1997 With Lagged FEDFUNDS (Q1)"
     ),

    # IV 1997 With Lagged FEDFUNDS (All Quarters)
    (results_1997_with_lagged_all,
     train_data[["FedFunds_1997_Lag1"]],  # Exogenous variables for training
     train_data[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     # Endogenous variables for training
     train_data["FEDFUNDS_19970107"],  # Dependent variable for training
     test_data[["FedFunds_1997_Lag1"]],  # Exogenous variables for testing
     test_data[["Inflation_Rate_1997", "OutputGap_1997", "HOUST_19970122"]],
     # Endogenous variables for testing
     test_data["FEDFUNDS_19970107"],  # Dependent variable for testing
     "IV 1997 With Lagged FEDFUNDS (All Quarters)"
     ),

    # IV 2002 Without Lagged FEDFUNDS (First Quarter Only)
    (results_2002_without_lagged_q1,
     train_data_q1[[]],  # Exogenous variables for training
     train_data_q1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     # Endogenous variables for training
     train_data_q1["FEDFUNDS_20020108"],  # Dependent variable for training
     test_data_q1[[]],  # Exogenous variables for testing
     test_data_q1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     # Endogenous variables for testing
     test_data_q1["FEDFUNDS_20020108"],  # Dependent variable for testing
     "IV 2002 Without Lagged FEDFUNDS (Q1)"
     ),

    # IV 2002 Without Lagged FEDFUNDS (All Quarters)
    (results_2002_without_lagged_all,
     train_data[[]],  # Exogenous variables for training
     train_data[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     # Endogenous variables for training
     train_data["FEDFUNDS_20020108"],  # Dependent variable for training
     test_data[[]],  # Exogenous variables for testing
     test_data[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     # Endogenous variables for testing
     test_data["FEDFUNDS_20020108"],  # Dependent variable for testing
     "IV 2002 Without Lagged FEDFUNDS (All Quarters)"
     ),

    # IV 2002 With Lagged FEDFUNDS (First Quarter Only)
    (results_2002_with_lagged_q1,
     train_data_q1[["FedFunds_2002_Lag1"]],  # Exogenous variables for training
     train_data_q1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     # Endogenous variables for training
     train_data_q1["FEDFUNDS_20020108"],  # Dependent variable for training
     test_data_q1[["FedFunds_2002_Lag1"]],  # Exogenous variables for testing
     test_data_q1[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     # Endogenous variables for testing
     test_data_q1["FEDFUNDS_20020108"],  # Dependent variable for testing
     "IV 2002 With Lagged FEDFUNDS (Q1)"
     ),

    # IV 2002 With Lagged FEDFUNDS (All Quarters)
    (results_2002_with_lagged_all,
     train_data[["FedFunds_2002_Lag1"]],  # Exogenous variables for training
     train_data[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     # Endogenous variables for training
     train_data["FEDFUNDS_20020108"],  # Dependent variable for training
     test_data[["FedFunds_2002_Lag1"]],  # Exogenous variables for testing
     test_data[["Inflation_Rate_2002", "OutputGap_2002", "HOUST_20020117"]],
     # Endogenous variables for testing
     test_data["FEDFUNDS_20020108"],  # Dependent variable for testing
     "IV 2002 With Lagged FEDFUNDS (All Quarters)"
     )
]

all_metrics = []


# Process OLS models
for model, X_train, y_train, X_test, y_test, model_name in ols_models:
    all_metrics.append(calculate_metrics(model, X_train, y_train, X_test, y_test, model_name))

# Process IV models
for model, X_train_exog, X_train_endog, y_train, X_test_exog, X_test_endog, y_test, model_name in iv_models:
    all_metrics.append(calculate_metrics_iv(model, X_train_exog, X_train_endog, y_train, X_test_exog, X_test_endog, y_test, model_name))

# Creating a DataFrame for the metrics
metrics_df = pd.DataFrame(all_metrics)

# Displaying the DataFrame
print(metrics_df)

print(merged_data.columns)

for ols in ols_models:
    print(ols[-1])
    print(ols[0].f_pvalue)

for iv in iv_models:
    print(iv[-1])
    print(iv[0].f_statistic.pval)


for iv in iv_models:
    print(iv[-1])
    print(iv[0].params)
    print(iv[0].pvalues)

for ols in ols_models:
    print(ols[-1])
    print(ols[0].params)
    print(ols[0].pvalues)