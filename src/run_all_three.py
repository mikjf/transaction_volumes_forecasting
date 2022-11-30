# STANDARD IMPORTS
import pandas as pd
import numpy as np
from numpy import array
import scipy
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math
import sklearn.metrics
from sklearn.metrics import mean_squared_error
import json

# MATPLOTLIB
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# PLOTLY
import plotly.express as px
import plotly.graph_objects as go

# ARIMA
from pmdarima.arima import auto_arima
import pickle

# ETS
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from statsmodels.iolib.smpickle import load_pickle

# PROPHET
from prophet import Prophet

# PY FILES ARIMA
from get_sum import get_sum
import auto_arima_single
from auto_arima_single import run_arima_get_rmse
from run_arima_tune_get_pred import run_arima
from auto_arima_single import calc_error

# PY FILES ETS
from ETS_Function import run_ETS_get_rmse
from ETS_Function import predictions

# PY FILES PROPHET
from auto_single_prophet import run_prophet_get_rmse
from run_prophet_tune_get_pred import run_prophet
from prophet.serialize import model_to_json, model_from_json

# VISUALIZATION
from Visualization import plot_simulation

###################################################################################

# FILE PATHS - YOU NEED TO CHANGE THIS
source = 'data/raw/Mock_totals.csv'
output = 'data/processed/'
output_for_trac = 'tracking_files/'
output_for_model = 'models/'

###################################################################################

# read file with the monthly totals
true_values = pd.read_csv(source)
# create timestamp format
dates = pd.Series(pd.period_range(true_values.Month[0], freq="M", periods=len(true_values.Month)))
# create df of the monthly transactions
data = pd.DataFrame({'Month':dates,'total':true_values.total.values})
# output
print('Total monthly transaction volumes for', len(data.index), 'months')
print(data)

###################################################################################

# MEASURE_RMSE FUNCTION

###################################################################################

models = ['ARIMA', 'ETS', 'PROPHET']
rmse_models = dict.fromkeys(models)

# RUN ARIMA get RMSE
rmse_models['ARIMA'] = (run_arima_get_rmse(merchant=data,merchant_name='total',train_test_split=21,seasonality=12,D_val=0))

# RUN ETS get RMSE
cfg, rmse_models['ETS'] =  run_ETS_get_rmse(data)

# RUN PROPHET get RMSE
rmse_models['PROPHET'] = (run_prophet_get_rmse(merchant=data, train_test_split = 21))

print('List of RMSE per model, train test error')
print(rmse_models)

# best = [arima, prophet, ets] depending on the smallest error
best = min(rmse_models, key=rmse_models.get)
print('The best model is {}'.format(best))

#############################################################################

rmse_models_fitted_vs_true = dict.fromkeys(models)

#############################################################################

# TRAIN ALL 3 MODELS, PREDICT 12 MONTHS USING ALL THREE

#############################################################################

# PROPHET

df_predictions_pr, df_fitted_pr, model_pr = run_prophet(data, pred_months=12)
df_fitted_pr.to_csv(output+str(data.Month.max())+'_Pro_fit.csv')
df_predictions_pr.to_csv(output+str(data.Month.max())+'_Pro_pred.csv')
rmse_models_fitted_vs_true['PROPHET'] = calc_error(data.total.values,df_fitted_pr.fitted.values,method='RMSE', normalized=False)

# tracking file for PROPHET
prophet_track = {}
prophet_track['path']=output_for_model+str(data.Month.max())+'_prophet_model.json'
prophet_track['predictions']=output+str(data.Month.max())+'_Pro_pred.csv'
prophet_track['type']='PROPHET'
prophet_track['rmse']=rmse_models_fitted_vs_true['PROPHET']
prophet_track['generation']=str(data.Month.max())
prophet_track_file = {str(data.Month.max())+'_Prophet': prophet_track}
with open(output_for_trac+str(data.Month.max())+'_Prophet_tracking.json', "w") as outfile:
    json.dump(prophet_track_file, outfile)

# saving model PROPHET
with open(output_for_model+str(data.Month.max())+'_prophet_model.json', 'w') as fout:
    fout.write(model_to_json(model_pr))

#############################################################################

# ETS

df_predictions_ets, df_fitted_ets, model_ets = predictions(data, cfg)
df_predictions_ets.to_csv(output+str(data.Month.max())+'_ETS_pred.csv')
df_fitted_ets.to_csv(output+str(data.Month.max())+'_ETS_fit.csv')
rmse_models_fitted_vs_true['ETS'] = calc_error(data.total.values,df_fitted_ets.fitted.values,method='RMSE', normalized=False)

# tracking file for ETS
ETS_track = {}
ETS_track['path']=output_for_model+str(data.Month.max())+'_ETS_model.pkl'
ETS_track['predictions']=output+str(data.Month.max())+'_ETS_pred.csv'
ETS_track['type']='ETS'
ETS_track['rmse']=rmse_models_fitted_vs_true['ETS']
ETS_track['generation']=str(data.Month.max())
ETS_track_file = {str(data.Month.max())+'_ETS': ETS_track}
with open(output_for_trac+str(data.Month.max())+'_ETS_tracking.json', "w") as outfile:
    json.dump(ETS_track_file, outfile)

# saving ETS model
model_ets.save(output_for_model+str(data.Month.max())+'_ETS_model.pkl')

#############################################################################

# ARIMA
predictions_df_ar, fitted_df_ar, model_ar = run_arima(data, 'total', seasonality=12, D_val=0, pred_months=12)
predictions_df_ar.to_csv(output+str(data.Month.max())+'_ari_pred.csv')
fitted_df_ar.to_csv(output+str(data.Month.max())+'_ari_fit.csv')
rmse_models_fitted_vs_true['ARIMA'] = calc_error(data.total.values[1:],fitted_df_ar.fitted.values[1:],method='RMSE', normalized=False)

# tracking file for ARIMA
arima_track = {}
arima_track['path']=output_for_model+str(data.Month.max())+'_arima_model.pkl'
arima_track['predictions']=output+str(data.Month.max())+'_ari_pred.csv'
arima_track['type']='ARIMA'
arima_track['rmse']=rmse_models_fitted_vs_true['ARIMA']
arima_track['generation']=str(data.Month.max())
arima_track_file = {str(data.Month.max())+'_ARIMA': arima_track}
with open(output_for_trac+str(data.Month.max())+'_ARIMA_tracking.json', "w") as outfile:
    json.dump(arima_track_file, outfile)

# saving model ARIMA
with open(output_for_model+str(data.Month.max())+'_arima_model.pkl', 'wb') as pkl:
    pickle.dump(model_ar, pkl)

# VISUALISATION - you can comment out, if you don't need the visualisation
plot_simulation(data, predictions_df_ar, fitted_df_ar, df_fitted_ets, df_predictions_ets, df_predictions_pr, df_fitted_pr)