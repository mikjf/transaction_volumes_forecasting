import pandas as pd
import numpy as np
import scipy
from datetime import datetime
from pmdarima.arima import auto_arima

def run_arima(merchant, merchant_name, seasonality = 12, D_val =1, pred_months = 12):
  # merchant is a df created from row of a data frame we read
  merchant = merchant.set_index('Month')
  merchant['month_index'] = merchant.index.month
  
  # fit the model
  model_final = auto_arima(merchant[[merchant_name]], exogenous=merchant[['month_index']],
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
                           
  # 12 month prediction
  
  # time frame for forecasting
  forecast_df = pd.DataFrame({"month_index":pd.date_range(merchant.index[-1], periods = pred_months, freq='MS').month},
                    index = pd.date_range(merchant.index[-1] + pd.DateOffset(months=1), periods = pred_months, freq='MS'))
  
  prediction, confint_pred = model_final.predict(n_periods=pred_months, 
                                        return_conf_int=True,
                                        exogenous=forecast_df[['month_index']])
  
  # save data
                                        
  CI_lower = [item[0] for item in confint_pred]
  CI_upper = [item[1] for item in confint_pred]
  #CI_width = CI_upper.values - CI_lower.values
  
  prediction.to_csv(file_path+'data_predictions.csv')
  CI_lower.to_csv(file_path+'CI_lower.csv')
  CI_upper.to_csv(file_path+'CI_upper.csv')
  
  return prediction, confint_pred
