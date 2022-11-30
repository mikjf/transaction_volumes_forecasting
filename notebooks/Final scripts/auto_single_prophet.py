# Prophet model for time series forecast
from prophet import Prophet

# Data processing
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Model performance evaluation
#from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def calc_error(y,y_hat,method='MAE', normalized=False):
  # y true values
  # y_hat predicted
  # method can be MAE or RMSE
  # normalized can be False, avarage, range, iqr
  if method == 'MAE':
    error_metric = np.mean((np.abs(y_hat - y)))
  else:
    error_metric = np.sqrt(np.mean((y_hat - y)**2))
  if normalized == False:
    return error_metric
  else:
    return normalize_error(y,error_metric,normalized)

def normalize_error(y,metric,method):
  # method = avarage, range, iqr
  if method == 'average':
    metric = metric/y.mean()
  elif method == 'range':
    metric = metric/(y.max()-y.min())
  else:
    metric = metric/scipy.stats.iqr(y)
  return metric

def run_prophet_get_rmse(merchant, train_test_split = 21):
  # merchant is a df created from row of a data frame we read
  # train_test_split - number of months for train
  
  data = pd.DataFrame(merchant.values)
# Change variable names
  my_input = ['ds', 'y'] 
  data.columns = my_input
  
  #train test split, since arima runs on min 21, we used that for train, and whatever is left for test
  start_date = data.ds[0]
  end_date = data.ds[len(data.ds)-1]
  train_end_date = data.ds[train_test_split]
  train = data[data['ds'] <= train_end_date]
  test = data[data['ds'] > train_end_date]
  
  # find best hyperparameters and fit the model
  model = Prophet(interval_width=0.99, seasonality_mode='multiplicative')
  
  # Fit the model on the training dataset
  model.fit(train)
  
  results = model.predict(pd.DataFrame({'ds':test.ds.values}))
  
  fitted = results.yhat
  
  rmse = calc_error(test.y.values, fitted.values,method = 'RMSE')
  #norm_rmse = calc_error(test[merchant_name].values, fitted.values,method = 'RMSE', normalized = 'range')
  
  return rmse
