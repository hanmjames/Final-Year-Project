import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
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

# Calculate the lagged Fed Funds rates for 1997 and 2002 before filtering
merged_data['FedFunds_Lag_1997'] = merged_data['FEDFUNDS_19970107'].shift(1)
merged_data['FedFunds_Lag_2002'] = merged_data['FEDFUNDS_20020108'].shift(1)

# Restrict the range of analysis to 1981-1996 (include all quarters)
merged_data = merged_data[(merged_data['observation_date'] >= '1981-01-01') & (merged_data['observation_date'] <= '1996-12-31')]

# Ensure 'observation_date' is retained in the merged data
merged_data.reset_index(drop=True, inplace=True)

# Display the updated merged data with output gaps and inflation rate
print("Merged Data (After Adding Output Gaps and Annualized Inflation Rate):")
print(merged_data.head())

# Filter the data to include only the first day of January for each year (16 points for analysis)
merged_data_q1 = merged_data.loc[merged_data['observation_date'].dt.strftime('%m-%d') == '01-01'].copy()

# Display merged_data_q1 to confirm the selection of Q1 only
print("Filtered Data for Q1 (First Day of Each Year):")
print(merged_data_q1)

# OLS Regression for 1997 Without Lagged FedFunds (First Quarter Only)
y_1997_q1 = merged_data_q1["FEDFUNDS_19970107"]
X_1997_q1 = merged_data_q1[["Inflation_Rate_1997", "OutputGap_1997"]]
X_1997_q1 = add_constant(X_1997_q1)

# Fit OLS model for Q1 only
ols_1997_without_lag_q1 = OLS(y_1997_q1, X_1997_q1).fit()

# Display the OLS results for the first quarter data
print("OLS Results for 1997 (Without Lagged FedFunds, First Quarter Only):")
print(ols_1997_without_lag_q1.summary())

# OLS Regression for 1997 Without Lagged FedFunds (All Quarters)
y_1997_all = merged_data["FEDFUNDS_19970107"]
X_1997_all = merged_data[["Inflation_Rate_1997", "OutputGap_1997"]]
X_1997_all = add_constant(X_1997_all)

# Fit OLS model for all quarters
ols_1997_without_lag_all = OLS(y_1997_all, X_1997_all).fit()

# Display the OLS results for all quarters data
print("OLS Results for 1997 (Without Lagged FedFunds, All Quarters):")
print(ols_1997_without_lag_all.summary())

# OLS Regression for 1997 With Lagged FedFunds (First Quarter Only)
y_1997_q1_lagged = merged_data_q1["FEDFUNDS_19970107"]
X_1997_q1_lagged = merged_data_q1[["FedFunds_Lag_1997", "Inflation_Rate_1997", "OutputGap_1997"]]
X_1997_q1_lagged = add_constant(X_1997_q1_lagged.dropna())
y_1997_q1_lagged = y_1997_q1_lagged.loc[X_1997_q1_lagged.index]

ols_1997_with_lag_q1 = OLS(y_1997_q1_lagged, X_1997_q1_lagged).fit()
print("OLS Results for 1997 (With Lagged FedFunds, First Quarter Only):")
print(ols_1997_with_lag_q1.summary())

# OLS Regression for 1997 With Lagged FedFunds (All Quarters)
y_1997_all_lagged = merged_data["FEDFUNDS_19970107"]
X_1997_all_lagged = merged_data[["FedFunds_Lag_1997", "Inflation_Rate_1997", "OutputGap_1997"]]
X_1997_all_lagged = add_constant(X_1997_all_lagged.dropna())
y_1997_all_lagged = y_1997_all_lagged.loc[X_1997_all_lagged.index]

ols_1997_with_lag_all = OLS(y_1997_all_lagged, X_1997_all_lagged).fit()
print("OLS Results for 1997 (With Lagged FedFunds, All Quarters):")
print(ols_1997_with_lag_all.summary())

# OLS Regression for 2002 Without Lagged FedFunds (First Quarter Only)
y_2002_q1 = merged_data_q1["FEDFUNDS_20020108"]
X_2002_q1 = merged_data_q1[["Inflation_Rate_2002", "OutputGap_2002"]]
X_2002_q1 = add_constant(X_2002_q1.dropna())
y_2002_q1 = y_2002_q1.loc[X_2002_q1.index]

ols_2002_without_lag_q1 = OLS(y_2002_q1, X_2002_q1).fit()
print("OLS Results for 2002 (Without Lagged FedFunds, First Quarter Only):")
print(ols_2002_without_lag_q1.summary())

# OLS Regression for 2002 Without Lagged FedFunds (All Quarters)
y_2002_all = merged_data["FEDFUNDS_20020108"]
X_2002_all = merged_data[["Inflation_Rate_2002", "OutputGap_2002"]]
X_2002_all = add_constant(X_2002_all.dropna())
y_2002_all = y_2002_all.loc[X_2002_all.index]

ols_2002_without_lag_all = OLS(y_2002_all, X_2002_all).fit()
print("OLS Results for 2002 (Without Lagged FedFunds, All Quarters):")
print(ols_2002_without_lag_all.summary())

# OLS Regression for 2002 With Lagged FedFunds (First Quarter Only)
y_2002_q1_lagged = merged_data_q1["FEDFUNDS_20020108"]
X_2002_q1_lagged = merged_data_q1[["FedFunds_Lag_2002", "Inflation_Rate_2002", "OutputGap_2002"]]
X_2002_q1_lagged = add_constant(X_2002_q1_lagged.dropna())
y_2002_q1_lagged = y_2002_q1_lagged.loc[X_2002_q1_lagged.index]

ols_2002_with_lag_q1 = OLS(y_2002_q1_lagged, X_2002_q1_lagged).fit()
print("OLS Results for 2002 (With Lagged FedFunds, First Quarter Only):")
print(ols_2002_with_lag_q1.summary())

# OLS Regression for 2002 With Lagged FedFunds (All Quarters)
y_2002_all_lagged = merged_data["FEDFUNDS_20020108"]
X_2002_all_lagged = merged_data[["FedFunds_Lag_2002", "Inflation_Rate_2002", "OutputGap_2002"]]
X_2002_all_lagged = add_constant(X_2002_all_lagged.dropna())
y_2002_all_lagged = y_2002_all_lagged.loc[X_2002_all_lagged.index]

ols_2002_with_lag_all = OLS(y_2002_all_lagged, X_2002_all_lagged).fit()
print("OLS Results for 2002 (With Lagged FedFunds, All Quarters):")
print(ols_2002_with_lag_all.summary())

def save_summary(summary, title, file_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0, 0, str(summary), fontsize=10, family="monospace")
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    plt.savefig(file_name, bbox_inches="tight")

# Save summaries for 1997 Without Lagged FedFunds
save_summary(ols_1997_without_lag_q1.summary(), "OLS Results for 1997 (Without Lagged FedFunds, First Quarter Only)", "ols_1997_without_lag_q1.png")
save_summary(ols_1997_without_lag_all.summary(), "OLS Results for 1997 (Without Lagged FedFunds, All Quarters)", "ols_1997_without_lag_all.png")

# Save summaries for 1997 With Lagged FedFunds
save_summary(ols_1997_with_lag_q1.summary(), "OLS Results for 1997 (With Lagged FedFunds, First Quarter Only)", "ols_1997_with_lag_q1.png")
save_summary(ols_1997_with_lag_all.summary(), "OLS Results for 1997 (With Lagged FedFunds, All Quarters)", "ols_1997_with_lag_all.png")

# Save summaries for 2002 Without Lagged FedFunds
save_summary(ols_2002_without_lag_q1.summary(), "OLS Results for 2002 (Without Lagged FedFunds, First Quarter Only)", "ols_2002_without_lag_q1.png")
save_summary(ols_2002_without_lag_all.summary(), "OLS Results for 2002 (Without Lagged FedFunds, All Quarters)", "ols_2002_without_lag_all.png")

# Save summaries for 2002 With Lagged FedFunds
save_summary(ols_2002_with_lag_q1.summary(), "OLS Results for 2002 (With Lagged FedFunds, First Quarter Only)", "ols_2002_with_lag_q1.png")
save_summary(ols_2002_with_lag_all.summary(), "OLS Results for 2002 (With Lagged FedFunds, All Quarters)", "ols_2002_with_lag_all.png")
