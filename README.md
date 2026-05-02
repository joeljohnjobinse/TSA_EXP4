# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 02/05/2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
#exp4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("Walmart_sales.csv")

print("Columns:", df.columns)

data = df['Weekly_Sales']

data = pd.to_numeric(data, errors='coerce')
data = data.dropna()

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    data = df['Weekly_Sales'].values
else:
    data = data.values

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plot_acf(data, lags=40, ax=plt.gca())
plt.title("Autocorrelation (ACF)")

plt.subplot(1,2,2)
plot_pacf(data, lags=40, ax=plt.gca())
plt.title("Partial Autocorrelation (PACF)")

plt.tight_layout()
plt.show()

model_11 = ARIMA(data, order=(1,0,1))
fit_11 = model_11.fit()

print("\nARMA(1,1) Summary:")
print(fit_11.summary())

plt.figure(figsize=(10,4))
plt.plot(data, label='Actual')
plt.plot(fit_11.fittedvalues, color='red', label='Fitted ARMA(1,1)')
plt.legend()
plt.title("ARMA(1,1) Fit")
plt.grid()
plt.show()


model_22 = ARIMA(data, order=(2,0,2))
fit_22 = model_22.fit()

print("\nARMA(2,2) Summary:")
print(fit_22.summary())

plt.figure(figsize=(10,4))
plt.plot(data, label='Actual')
plt.plot(fit_22.fittedvalues, color='green', label='Fitted ARMA(2,2)')
plt.legend()
plt.title("ARMA(2,2) Fit")
plt.grid()
plt.show()

### OUTPUT:

<img width="873" height="520" alt="image" src="https://github.com/user-attachments/assets/84afc942-73db-4b34-9cea-4950e2fb587e" />

SIMULATED ARMA(1,1) PROCESS:




Partial Autocorrelation
<img width="407" height="331" alt="image" src="https://github.com/user-attachments/assets/36eedf5f-9e90-44a3-a84e-02830e97f4cb" />

Autocorrelation
<img width="428" height="347" alt="image" src="https://github.com/user-attachments/assets/cb5f213e-462a-4212-8bf0-2a0c18792ea0" />



SIMULATED ARMA(2,2) PROCESS:
<img width="815" height="316" alt="image" src="https://github.com/user-attachments/assets/ad61469a-66e4-4398-bab7-075301cf911e" />


SARIMAX:
<img width="846" height="508" alt="image" src="https://github.com/user-attachments/assets/020d9db2-17d0-49bc-bcbf-16e5cae23033" />


<img width="843" height="518" alt="image" src="https://github.com/user-attachments/assets/e4c33f7a-19ef-4401-b358-d695ae00981e" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
