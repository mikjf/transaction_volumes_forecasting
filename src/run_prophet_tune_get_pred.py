# Prophet model for time series forecast
# IMPORTS
from prophet import Prophet

# DATA PROCESSING
import numpy as np
import pandas as pd

# VISUALIZATION
import seaborn as sns
import matplotlib.pyplot as plt

###################################################################################

# RUN_PROPHET FUNCTION

def run_prophet(merchant, pred_months = 12):

  # merchant is a df created from row of a data frame we read
  data = merchant.copy(deep=True)
  # change variable names
  my_input = ['ds', 'y'] 
  data.columns = my_input
  data['ds'] = data['ds'].dt.strftime('%Y-%m')
  # find best hyperparameters and fit the model
  model = Prophet(interval_width=0.99, seasonality_mode='multiplicative')
  
  # fit the model on the training dataset
  model.fit(data)

###################################################################################

  # 12 MONTHS PREDICTION
  
  # time frame for forecasting
  forecast_df = pd.DataFrame({'ds':pd.date_range(start=data.ds[0], periods =len(data.index)+pred_months, freq="M")})
  results = model.predict(forecast_df)
  results['ds'] = results['ds'].dt.strftime('%Y-%m')
  results = results[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
  columns_pr = ['timestamp', 'predicted','lower_bound', 'upper_bound']
  results.columns = columns_pr
  results = results.set_index('timestamp')
  df_fitted = results.iloc[:-pred_months]
  df_fitted.rename(columns={'predicted':'fitted'}, inplace=True)
  df_predictions = results.iloc[-pred_months:]

  return df_predictions, df_fitted, model