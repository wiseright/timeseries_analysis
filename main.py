import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
plt.style.use('fivethirtyeight')
import matplotlib
matplotlib.use('TkAgg')

# Open the time series of Amazon stock price
amazon = pd.read_csv('amazon_close.csv',
                     index_col='date',
                     parse_dates=True)

amazon.plot()
plt.title('Amazon Stock Price vs Time')
plt.ylabel('Stock price[$]')

# Null Hypothesis is: time series is not stationary due to trends
# Use Adfuller test
#   The first element of this test is test statistic: The more negative this number is, the more likely that the data is stationary
#   The second element of this test is the p-value: if the p-value is smaller than 0.05, we reject the null hypothesis and assume our time series must be stationary
result = adfuller(amazon)
print(f'Test statistics: {result[0]}')      # -1.344
print(f'p-value: {result[1]}')              #  0.608 > 0.05 --> NOT STATIONARY

# To eliminate the trends we'll use the discrete derivative of the signal
amazon_diff = amazon.diff()
amazon_diff.dropna(inplace=True)

# Now we can consider the data stationary
result = adfuller(amazon_diff)
print(f'Test statistics: {result[0]}')
print(f'p-value: {result[1]}')

# Fit the model ARIMA (Auto-Regressive Integration Moving Average)
model = ARIMA(amazon_diff, order=(1, 1, 1)) # (p, d, q) -> p: autoregressive order, d: differencing order, q: moving average order
results = model.fit()
results.summary()


# Generate predictions
one_step_forecast = results.get_prediction(end=20)

# Extract prediction mean
mean_forecast = one_step_forecast.predicted_mean

# Get confidence intervals of  predictions
confidence_intervals = one_step_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower close']
upper_limits = confidence_intervals.loc[:,'upper close']

# Print best estimate  predictions
print(mean_forecast)


