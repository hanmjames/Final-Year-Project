import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant

realGDP = pd.read_csv(r"Training_and_Testing_Data/RealGDPTotal.csv")
potGDP = pd.read_csv(r"Training_and_Testing_Data/PotentialGDPTotal.csv")
inflation = pd.read_csv(r"Training_and_Testing_Data/InflationTotal.csv")
fedFunds = pd.read_csv(r"Training_and_Testing_Data/FedfundsTotal.csv")

mergedData = (
    realGDP.merge(potGDP, on="observation_date")
    .merge(inflation, on="observation_date")
    .merge(fedFunds, on="observation_date")
)

mergedData['observation_date'] = pd.to_datetime(mergedData['observation_date'])
mergedData['Inflation_Rate_1997'] = (
    (mergedData['GDPDEF'] / mergedData['GDPDEF'].shift(4) - 1) * 100
)
mergedData['OutputGap_1997'] = 100 * (
    np.log(mergedData['GDPC1']) - np.log(mergedData['GDPPOT'])
)
for lag in range(1, 4):
    mergedData[f'FedFunds_1997_Lag{lag}'] = mergedData['FEDFUNDS'].shift(lag)
    mergedData[f'OutputGap_1997_Lag{lag}'] = mergedData['OutputGap_1997'].shift(lag)
    mergedData[f'Inflation_Rate_1997_Lag{lag}'] = mergedData['Inflation_Rate_1997'].shift(lag)

mergedData = mergedData[(mergedData['observation_date'] >= '1981-01-01') & (mergedData['observation_date'] <= '2013-12-31')]
mergedData.dropna(subset=["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"], inplace=True)
mergedData.reset_index(drop=True, inplace=True)

# def growingWindowCoeffPlot(data, yCol, xCols, startIndex=24, title=None):
#     coefs = {col: [] for col in xCols}
#     dates = []
#
#     for i in range(startIndex, len(data)):
#         window = data.iloc[:i]
#         X = add_constant(window[xCols])
#         y = window[yCol]
#         model = OLS(y, X).fit()
#         for col in xCols:
#             coefs[col].append(model.params[col])
#         dates.append(data.iloc[i].observation_date)
#
#     plt.figure(figsize=(12, 6))
#     for col, values in coefs.items():
#         plt.plot(dates, values, label=f"{col} Coef")
#
#     plt.axvline(pd.to_datetime("2008-01-01"), color="red", linestyle="--", label="2008")
#     plt.title(title if title else "Growing Window OLS Coefficients (1981–2013)")
#     plt.xlabel("Date")
#     plt.ylabel("Coefficient")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def rollingCoeffPlot(data, yCol, xCols, windowSize=40, smooth=4, title=None):
    from statsmodels.api import OLS, add_constant
    import matplotlib.pyplot as plt
    import pandas as pd

    results = []
    for i in range(windowSize, len(data)):
        window = data.iloc[i - windowSize:i]
        if window[xCols + [yCol]].isnull().values.any():
            continue
        X = add_constant(window[xCols])
        y = window[yCol]
        try:
            model = OLS(y, X).fit()
            results.append({
                "Date": data.iloc[i].observation_date,
                **{col: model.params[col] for col in xCols}
            })
        except:
            continue

    dfCoeffs = pd.DataFrame(results).set_index("Date")
    dfCoeffs = dfCoeffs.rolling(smooth).mean()  # smooth the lines

    plt.figure(figsize=(12, 6))
    dfCoeffs.plot(ax=plt.gca())
    plt.axvline(pd.to_datetime("2008-01-01"), color="red", linestyle="--", label="2008")
    plt.title(title if title else f"Rolling OLS Coefficients ({windowSize}-Quarter Window, Smoothed)")
    plt.xlabel("Date")
    plt.ylabel("Coefficient")
    plt.legend()
    plt.tight_layout()
    plt.show()


rollingCoeffPlot(
    mergedData,
    yCol="FEDFUNDS",
    xCols=["FedFunds_1997_Lag1", "Inflation_Rate_1997", "OutputGap_1997"],
    windowSize=24,
    title="Rolling OLS Coefficients (24-Quarter Window, 1981–2013)"
)
