import pandas as pd
import numpy as np
import scipy
from datetime import datetime
from pmdarima.arima import auto_arima

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

def run_arima_get_rmse(merchant, merchant_name, train_test_split = 21, seasonality = 12, D_val =1):
  # merchant is a df created from row of a data frame we read
  # train_test_split - number of months for train
  # seasonality - what should the m be, based on the data
  # D_val - starting value for differencing the seasonal part of the time series
  merchant = merchant.set_index('Month')
  merchant['month_index'] = merchant.index.month
  
  #train test split, since arima runs on min 21, we used that for train, and whatever is left for test
  train = pd.DataFrame({merchant_name : merchant[merchant_name][:train_test_split].values,
                      'Month' : merchant[merchant_name][:train_test_split].index})
  train = train.set_index('Month')
  train['month_index'] = train.index.month
  
  test = pd.DataFrame({merchant_name : merchant[merchant_name][train_test_split:].values,
                      'Month' : merchant[merchant_name][train_test_split:].index})
  test = test.set_index('Month')
  test['month_index'] = test.index.month
  
  # find best hyperparameters and fit the model
  model = auto_arima(train[[merchant_name]], exogenous=train[['month_index']],
                   start_p=1, 
                   max_p=3,
                   start_q=1, 
                   max_q=3,
                   d=None,
                   max_d=3,
                   start_P=1,
                   max_P=3,
                   start_Q=1,
                   max_Q=3,
                   D=D_val,
                   max_D=3,
                   m=seasonality,  
                   seasonal=True,
                   test='adf',
                   trace=False,
                   error_action='ignore',  
                   suppress_warnings=True, 
                   stepwise=True)
  
  # validation: predict on the test data
  n_periods = len(test)
  df = train.copy(deep=True)
  
  fitted, confint = model.predict(n_periods=n_periods, 
                                  return_conf_int=True,
                                  exogenous=test[['month_index']])
  
  rmse = calc_error(test[merchant_name].values, fitted,method = 'RMSE')
  #norm_rmse = calc_error(test[merchant_name].values, fitted.values,method = 'RMSE', normalized = 'range')
  
  return rmse
