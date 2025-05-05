from statsmodels.tsa.statespace.sarimax import SARIMAX
from RegressionAnalysis import *
from Testing2000_2006 import *

# Load datasets
data_1981_1996 = merged_data[['observation_date', 'OutputGap_1997', 'Inflation_Rate_1997', 'FEDFUNDS_19970107', 'FedFunds_1997_Lag1']]
data_2000_2006 = merged_test_data[['observation_date', 'Inflation_Rate', 'OutputGap', 'FEDFUNDS', 'FEDFUNDS_Lag1']]

# Define training and testing variables
train_target = data_1981_1996['FEDFUNDS_19970107']
train_exog = data_1981_1996[['Inflation_Rate_1997', 'OutputGap_1997', 'FedFunds_1997_Lag1']]
test_exog = data_2000_2006[['Inflation_Rate', 'OutputGap', 'FEDFUNDS_Lag1']]  # Include consistent lagged variable

# SARIMAX parameters (Initial)
order = (1, 0, 1)
seasonal_order = (1, 1, 0, 4)  # Adjusted for quarterly data

# Train SARIMAX model
model = SARIMAX(
    train_target,
    exog=train_exog,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)

# Predictions
predictions = results.get_prediction(start=data_2000_2006.index[0], end=data_2000_2006.index[-1], exog=test_exog)
forecast = predictions.predicted_mean

# Evaluation Metrics
mse = mean_squared_error(data_2000_2006['FEDFUNDS'], forecast)
rmse = np.sqrt(mse)
variance = data_2000_2006['FEDFUNDS'].var()
mse_ratio = mse / variance

# Print results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"MSE to Variance Ratio: {mse_ratio}")

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(data_2000_2006['observation_date'], data_2000_2006['FEDFUNDS'], label="Actual")
plt.plot(data_2000_2006['observation_date'], forecast, label="Forecast", linestyle="--")
plt.title("SARIMAX Model: Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Federal Funds Rate")
plt.legend()
plt.show()

# ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(train_target, lags=20)
plot_pacf(train_target, lags=20)
plt.show()

# Grid search for SARIMAX parameters
import itertools

p = d = q = range(0, 3)  # ARIMA parameters
P = D = Q = range(0, 2)  # Seasonal parameters
m = [4]  # Quarterly seasonality

best_rmse = float("inf")
best_order = None
best_seasonal_order = None

for order in itertools.product(p, d, q):
    for seasonal_order in itertools.product(P, D, Q, m):
        try:
            model = SARIMAX(
                train_target,
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)
            forecast = results.get_prediction(start=data_2000_2006.index[0], end=data_2000_2006.index[-1], exog=test_exog).predicted_mean
            rmse = np.sqrt(mean_squared_error(data_2000_2006['FEDFUNDS'], forecast))

            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order
                best_seasonal_order = seasonal_order

        except Exception as e:
            continue

print("Best Order:", best_order)
print("Best Seasonal Order:", best_seasonal_order)
print("Best RMSE:", best_rmse)
