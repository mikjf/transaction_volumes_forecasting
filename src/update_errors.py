# IMPORTS
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
import pandas as pd
from auto_arima_single import calc_error

#############################################################################

# THIS YOU NEED TO CHANGE
source = 'data/raw/Mock_totals_new_moth_for_error_calc_test.csv'
output = 'data/processed/'
output_for_trac = 'tracking_files/'
output_for_model = 'models/'
output_validation_track = 'data/all_errors.csv'

#############################################################################

# load the true values
df_total = pd.read_csv(source)
df_total['timestamp'] = pd.to_datetime(df_total.Month).dt.to_period('m')

# get current month = last month in the dataset
month_current = str(df_total['timestamp'].iat[-1])

# get list of all the files from the directory
mypath = output
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# creating list of all the models
all_months=[]
for file in onlyfiles:
    all_months.append(file[:11])
models = (list(set(all_months)))

# load df with the predictions
predict_dfs = {}
for model in models:
     predict_dfs[model] = pd.read_csv(output+str(model)+'_pred.csv')

predict_for_err = {}
# get the predicted and true values for the period given by the dates for each model
for model in models:
    pred_df = predict_dfs[model]

    # create range of dates from earliest possible given the name of the model to the current date
    dates = pd.Series(
        pd.period_range(datetime.strptime(model[:7], '%Y-%m'), datetime.strptime(month_current, '%Y-%m'),
                        freq="M"))
    # dropping the first one, since we do not have pred for the last month of training
    dates_temp = dates.copy(deep=True).drop(labels=0)

    # get the true values for the period given by the dates
    true_temp = []
    for date in dates_temp:
        true_temp.append(df_total.loc[df_total['timestamp'] == date, 'total'].iloc[0])

    # get the predicted values
    temp_list = []
    for date in dates_temp:
        temp_list.append(pred_df.loc[pred_df['timestamp'] == str(date), 'predicted'].iloc[0])

    #create data frame for each model, and save to dictionary
    temp_df = pd.DataFrame({'timestamp':dates_temp, 'prediction':temp_list, 'true':true_temp})
    predict_for_err[model]=temp_df

# calculate the errors
errors_dict = {}

for model in models:
    df = predict_for_err[model]
    y = df['true'].values
    y_hat = df['prediction'].values
    errors_dict[model] = calc_error(y,y_hat,method='RMSE', normalized=False)
    # errors_dict gives us one column in the tracking files for the errors

# creating dataframe, so the column of errors can be saved
results = pd.DataFrame(errors_dict.items(), columns=['model', str(month_current)])

# checks, if the file already exists
# if yes it is updated with the new results column
# if not, a file is created
if os.path.isfile(output_validation_track):
    old = pd.read_csv(output_validation_track)
    df = pd.merge(old, results, on="model", how="right")
    updated_results = df.sort_values(by='model').drop(columns='Unnamed: 0')
    updated_results.to_csv(output_validation_track)
else:
    results.to_csv(output_validation_track)