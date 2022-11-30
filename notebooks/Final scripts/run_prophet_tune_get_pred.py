# Prophet model for time series forecast
from prophet import Prophet

# Data processing
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

def run_prophet(merchant, pred_months = 12):
  # merchant is a df created from row of a data frame we read
  data = pd.DataFrame(merchant.values)
# Change variable names
  my_input = ['ds', 'y'] 
  data.columns = my_input
  
  # find best hyperparameters and fit the model
  model = Prophet(interval_width=0.99, seasonality_mode='multiplicative')
  
  # Fit the model on the training dataset
  model.fit(train)
  
  # 12 month prediction
  
  # time frame for forecasting
  forecast_df = pd.DataFrame({'ds':pd.date_range(start=data.ds[len(data.ds)-1], periods = 12, freq="M")})
  
  results = model.predict(forecast_df)
  
  prediction = results[['ds','yhat']].set_index('ds')
  
  # save data
                                        
  CI_lower = results[['ds','y_lower']].set_index('ds')
  CI_upper = results[['ds','y_upper']].set_index('ds')
  #CI_width = CI_upper.values - CI_lower.values
  
  prediction.to_csv(file_path+'data_predictions.csv')
  CI_lower.to_csv(file_path+'CI_lower.csv')
  CI_upper.to_csv(file_path+'CI_upper.csv')
  
  return prediction, CI_lower, CI_upper
