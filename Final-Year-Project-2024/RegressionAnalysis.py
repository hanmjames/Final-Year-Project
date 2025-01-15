import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt

# Set pandas display option to show all columns
pd.set_option('display.max_columns', None)

# Load the separate datasets for real GDP, potential GDP, inflation, and federal funds rate
real_gdp = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\RealGDP_1981to1996.csv")
pot_gdp = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\PotGDP_1981to1996.csv")
inflation = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\Inflation_1981to1996.csv")
fed_funds = pd.read_csv(r"C:\Users\hanma\Documents\GitHub\Final-Year-Project\Final-Year-Project-2024\Fedfunds_1981to1996.csv")

# Merge datasets on 'observation_date'
merged_data = (
    real_gdp.merge(pot_gdp, on="observation_date")
    .merge(inflation, on="observation_date")
    .merge(fed_funds, on="observation_date")
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
    merged_data[f'OutputGap_2002_Lag{lag}'] = merged_data['OutputGap_2002'].shift(lag)
    merged_data[f'Inflation_Rate_2002_Lag{lag}'] = merged_data['Inflation_Rate_2002'].shift(lag)
    merged_data[f'FedFunds_2002_Lag{lag}'] = merged_data['FEDFUNDS_20020108'].shift(lag)

# Restrict the range of analysis to 1981-1996 (include all quarters)
merged_data = merged_data[(merged_data['observation_date'] >= '1981-01-01') & (merged_data['observation_date'] <= '1996-12-31')]

# Ensure 'observation_date' is retained in the merged data
merged_data.reset_index(drop=True, inplace=True)

# Display the updated merged data with output gaps and inflation rate
print("Merged Data (After Adding Output Gaps, Annualized Inflation Rate, and Lags):")
print(merged_data.head())

# Filter the data to include only the first day of January for each year (16 points for analysis)
merged_data_q1 = merged_data.loc[merged_data['observation_date'].dt.strftime('%m-%d') == '01-01'].copy()

# Display merged_data_q1 to confirm the selection of Q1 only
print("Filtered Data for Q1 (First Day of Each Year):")
print(merged_data_q1)

# Function to save regression summary as image
def save_summary(summary, title, file_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0, 0, str(summary), fontsize=10, family="monospace")
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    plt.savefig(file_name, bbox_inches="tight")

# OLS Regressions
# OLS Regression for 1997 Without Lagged FedFunds (First Quarter Only)
y_1997_q1 = merged_data_q1["FEDFUNDS_19970107"]
X_1997_q1 = merged_data_q1[["Inflation_Rate_1997", "OutputGap_1997"]]
X_1997_q1 = add_constant(X_1997_q1)

ols_1997_without_lag_q1 = OLS(y_1997_q1, X_1997_q1).fit()
print("OLS Results for 1997 (Without Lagged FedFunds, First Quarter Only):")
print(ols_1997_without_lag_q1.summary())

# OLS Regression for 1997 Without Lagged FedFunds (All Quarters)
y_1997_all = merged_data["FEDFUNDS_19970107"]
X_1997_all = merged_data[["Inflation_Rate_1997", "OutputGap_1997"]]
X_1997_all = add_constant(X_1997_all)

ols_1997_without_lag_all = OLS(y_1997_all, X_1997_all).fit()
print("OLS Results for 1997 (Without Lagged FedFunds, All Quarters):")
print(ols_1997_without_lag_all.summary())

# OLS Regression for 1997 With Lagged FedFunds (First Quarter Only)
y_1997_q1_lagged = merged_data_q1["FEDFUNDS_19970107"]
X_1997_q1_lagged = add_constant(merged_data_q1[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]])

ols_1997_with_lag_q1 = OLS(y_1997_q1_lagged, X_1997_q1_lagged).fit()
print("OLS Results for 1997 (With Lagged FedFunds, First Quarter Only):")
print(ols_1997_with_lag_q1.summary())

# OLS Regression for 1997 With Lagged FedFunds (All Quarters)
y_1997_all_lagged = merged_data["FEDFUNDS_19970107"]
X_1997_all_lagged = add_constant(merged_data[["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"]])

ols_1997_with_lag_all = OLS(y_1997_all_lagged, X_1997_all_lagged).fit()
print("OLS Results for 1997 (With Lagged FedFunds, All Quarters):")
print(ols_1997_with_lag_all.summary())

print(merged_data_q1[["OutputGap_1997_Lag1", "Inflation_Rate_1997_Lag1"]].corr())
print(merged_data_q1[["OutputGap_1997_Lag1", "Inflation_Rate_1997_Lag1"]].isnull().sum())


# IV Regressions
# IV Regression for 1997 Without Lagged FedFunds (First Quarter Only)
dependent_1997_q1 = merged_data_q1["FEDFUNDS_19970107"]
exog_1997_q1 = add_constant(merged_data_q1[[]])  # Only the constant term remains
endog_1997_q1 = merged_data_q1[["Inflation_Rate_1997", "OutputGap_1997"]]
instruments_1997_q1 = merged_data_q1[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3"
]]
iv_model_1997_q1 = IV2SLS(dependent_1997_q1, exog_1997_q1, endog_1997_q1, instruments_1997_q1)
results_1997_without_lagged_q1 = iv_model_1997_q1.fit()
print("IV Regression Results for 1997 Vintage Excl. Lagged FEDFUNDS (First Quarter Only):")
print(results_1997_without_lagged_q1.summary)

# IV Regression for 1997 Without Lagged FedFunds (All Quarters)
dependent_1997_all = merged_data["FEDFUNDS_19970107"]
exog_1997_all = add_constant(merged_data[[]])  # Only the constant term remains
endog_1997_all = merged_data[["Inflation_Rate_1997", "OutputGap_1997"]]
instruments_1997_all = merged_data[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3"
]]
iv_model_1997_all = IV2SLS(dependent_1997_all, exog_1997_all, endog_1997_all, instruments_1997_all)
results_1997_without_lagged_all = iv_model_1997_all.fit()
print("IV Regression Results for 1997 Vintage Excl. Lagged FEDFUNDS (All Quarters):")
print(results_1997_without_lagged_all.summary)

# IV Regression for 1997 With Lagged FedFunds (First Quarter Only)
dependent_1997_q1_lagged = merged_data_q1["FEDFUNDS_19970107"]
exog_1997_q1_lagged = add_constant(merged_data_q1[["FedFunds_1997_Lag1"]])
endog_1997_q1_lagged = merged_data_q1[["Inflation_Rate_1997", "OutputGap_1997"]]
instruments_1997_q1_lagged = merged_data_q1[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3"
]]
iv_model_1997_q1_lagged = IV2SLS(dependent_1997_q1_lagged, exog_1997_q1_lagged, endog_1997_q1_lagged, instruments_1997_q1_lagged)
results_1997_with_lagged_q1 = iv_model_1997_q1_lagged.fit()
print("IV Regression Results for 1997 Vintage Incl. Lagged FEDFUNDS (First Quarter Only):")
print(results_1997_with_lagged_q1.summary)

# IV Regression for 1997 With Lagged FedFunds (All Quarters)
dependent_1997_all_lagged = merged_data["FEDFUNDS_19970107"]
exog_1997_all_lagged = add_constant(merged_data[["FedFunds_1997_Lag1"]])
endog_1997_all_lagged = merged_data[["Inflation_Rate_1997", "OutputGap_1997"]]
instruments_1997_all_lagged = merged_data[[
    "OutputGap_1997_Lag1", "OutputGap_1997_Lag2", "OutputGap_1997_Lag3",
    "Inflation_Rate_1997_Lag1", "Inflation_Rate_1997_Lag2", "Inflation_Rate_1997_Lag3"
]]
iv_model_1997_all_lagged = IV2SLS(dependent_1997_all_lagged, exog_1997_all_lagged, endog_1997_all_lagged, instruments_1997_all_lagged)
results_1997_with_lagged_all = iv_model_1997_all_lagged.fit()
print("IV Regression Results for 1997 Vintage Incl. Lagged FEDFUNDS (All Quarters):")
print(results_1997_with_lagged_all.summary)

# IV Regression for 2002 Without Lagged FedFunds (First Quarter Only)
dependent_2002_q1 = merged_data_q1["FEDFUNDS_20020108"]
exog_2002_q1 = add_constant(merged_data_q1[[]])  # Only the constant term remains
endog_2002_q1 = merged_data_q1[["Inflation_Rate_2002", "OutputGap_2002"]]
instruments_2002_q1 = merged_data_q1[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3"
]]
iv_model_2002_q1 = IV2SLS(dependent_2002_q1, exog_2002_q1, endog_2002_q1, instruments_2002_q1)
results_2002_without_lagged_q1 = iv_model_2002_q1.fit()
print("IV Regression Results for 2002 Vintage Excl. Lagged FEDFUNDS (First Quarter Only):")
print(results_2002_without_lagged_q1.summary)

# IV Regression for 2002 Without Lagged FedFunds (All Quarters)
dependent_2002_all = merged_data["FEDFUNDS_20020108"]
exog_2002_all = add_constant(merged_data[[]])  # Only the constant term remains
endog_2002_all = merged_data[["Inflation_Rate_2002", "OutputGap_2002"]]
instruments_2002_all = merged_data[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3"
]]
iv_model_2002_all = IV2SLS(dependent_2002_all, exog_2002_all, endog_2002_all, instruments_2002_all)
results_2002_without_lagged_all = iv_model_2002_all.fit()
print("IV Regression Results for 2002 Vintage Excl. Lagged FEDFUNDS (All Quarters):")
print(results_2002_without_lagged_all.summary)

# IV Regression for 2002 With Lagged FedFunds (First Quarter Only)
dependent_2002_q1_lagged = merged_data_q1["FEDFUNDS_20020108"]
exog_2002_q1_lagged = add_constant(merged_data_q1[["FedFunds_2002_Lag1"]])
endog_2002_q1_lagged = merged_data_q1[["Inflation_Rate_2002", "OutputGap_2002"]]
instruments_2002_q1_lagged = merged_data_q1[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3"
]]
iv_model_2002_q1_lagged = IV2SLS(dependent_2002_q1_lagged, exog_2002_q1_lagged, endog_2002_q1_lagged, instruments_2002_q1_lagged)
results_2002_with_lagged_q1 = iv_model_2002_q1_lagged.fit()
print("IV Regression Results for 2002 Vintage Incl. Lagged FEDFUNDS (First Quarter Only):")
print(results_2002_with_lagged_q1.summary)

# IV Regression for 2002 With Lagged FedFunds (All Quarters)
dependent_2002_all_lagged = merged_data["FEDFUNDS_20020108"]
exog_2002_all_lagged = add_constant(merged_data[["FedFunds_2002_Lag1"]])
endog_2002_all_lagged = merged_data[["Inflation_Rate_2002", "OutputGap_2002"]]
instruments_2002_all_lagged = merged_data[[
    "OutputGap_2002_Lag1", "OutputGap_2002_Lag2", "OutputGap_2002_Lag3",
    "Inflation_Rate_2002_Lag1", "Inflation_Rate_2002_Lag2", "Inflation_Rate_2002_Lag3"
]]
iv_model_2002_all_lagged = IV2SLS(dependent_2002_all_lagged, exog_2002_all_lagged, endog_2002_all_lagged, instruments_2002_all_lagged)
results_2002_with_lagged_all = iv_model_2002_all_lagged.fit()
print("IV Regression Results for 2002 Vintage Incl. Lagged FEDFUNDS (All Quarters):")
print(results_2002_with_lagged_all.summary)


# Save summaries as images
# save_summary(results_1997_without_lagged_q1.summary, "IV Regression for 1997 Without Lagged FedFunds (First Quarter Only)", "iv_1997_without_lag_q1.png")
# save_summary(results_1997_without_lagged_all.summary, "IV Regression for 1997 Without Lagged FedFunds (All Quarters)", "iv_1997_without_lag_all.png")
# save_summary(results_1997_with_lagged_q1.summary, "IV Regression for 1997 With Lagged FedFunds (First Quarter Only)", "iv_1997_with_lag_q1.png")
# save_summary(results_1997_with_lagged_all.summary, "IV Regression for 1997 With Lagged FedFunds (All Quarters)", "iv_1997_with_lag_all.png")
# save_summary(results_2002_without_lagged_q1.summary, "IV Regression for 2002 Without Lagged FedFunds (First Quarter Only)", "iv_2002_without_lag_q1.png")
# save_summary(results_2002_without_lagged_all.summary, "IV Regression for 2002 Without Lagged FedFunds (All Quarters)", "iv_2002_without_lag_all.png")
# save_summary(results_2002_with_lagged_q1.summary, "IV Regression for 2002 With Lagged FedFunds (First Quarter Only)", "iv_2002_with_lag_q1.png")
# save_summary(results_2002_with_lagged_all.summary, "IV Regression for 2002 With Lagged FedFunds (All Quarters)", "iv_2002_with_lag_all.png")

# Extracting first-stage results for 1997 Without Lagged FedFunds (First Quarter Only)
print("First-Stage Results for 1997 Without Lagged FedFunds (First Quarter Only):")
print(results_1997_without_lagged_q1.first_stage)

# Extracting first-stage results for 1997 Without Lagged FedFunds (All Quarters)
print("First-Stage Results for 1997 Without Lagged FedFunds (All Quarters):")
print(results_1997_without_lagged_all.first_stage)

# Extracting first-stage results for 1997 With Lagged FedFunds (First Quarter Only)
print("First-Stage Results for 1997 With Lagged FedFunds (First Quarter Only):")
print(results_1997_with_lagged_q1.first_stage)

# Extracting first-stage results for 1997 With Lagged FedFunds (All Quarters)
print("First-Stage Results for 1997 With Lagged FedFunds (All Quarters):")
print(results_1997_with_lagged_all.first_stage)

# Extracting first-stage results for 2002 Without Lagged FedFunds (First Quarter Only)
print("First-Stage Results for 2002 Without Lagged FedFunds (First Quarter Only):")
print(results_2002_without_lagged_q1.first_stage)

# Extracting first-stage results for 2002 Without Lagged FedFunds (All Quarters)
print("First-Stage Results for 2002 Without Lagged FedFunds (All Quarters):")
print(results_2002_without_lagged_all.first_stage)

# Extracting first-stage results for 2002 With Lagged FedFunds (First Quarter Only)
print("First-Stage Results for 2002 With Lagged FedFunds (First Quarter Only):")
print(results_2002_with_lagged_q1.first_stage)

# Extracting first-stage results for 2002 With Lagged FedFunds (All Quarters)
print("First-Stage Results for 2002 With Lagged FedFunds (All Quarters):")
print(results_2002_with_lagged_all.first_stage)

# save_summary(results_1997_without_lagged_q1.first_stage, "First-Stage Results for 1997 Without Lagged FedFunds (First Quarter Only)", "first_stage_1997_without_lag_q1.png")
#
# save_summary(results_1997_without_lagged_all.first_stage, "First-Stage Results for 1997 Without Lagged FedFunds (All Quarters)", "first_stage_1997_without_lag_all.png")
#
# save_summary(results_1997_with_lagged_q1.first_stage, "First-Stage Results for 1997 With Lagged FedFunds (First Quarter Only)", "first_stage_1997_with_lag_q1.png")
#
# save_summary(results_1997_with_lagged_all.first_stage, "First-Stage Results for 1997 With Lagged FedFunds (All Quarters)", "first_stage_1997_with_lag_all.png")
#
# save_summary(results_2002_without_lagged_q1.first_stage, "First-Stage Results for 2002 Without Lagged FedFunds (First Quarter Only)", "first_stage_2002_without_lag_q1.png")
#
# save_summary(results_2002_without_lagged_all.first_stage, "First-Stage Results for 2002 Without Lagged FedFunds (All Quarters)", "first_stage_2002_without_lag_all.png")
#
# save_summary(results_2002_with_lagged_q1.first_stage, "First-Stage Results for 2002 With Lagged FedFunds (First Quarter Only)", "first_stage_2002_with_lag_q1.png")
#
# save_summary(results_2002_with_lagged_all.first_stage, "First-Stage Results for 2002 With Lagged FedFunds (All Quarters)", "first_stage_2002_with_lag_all.png")

