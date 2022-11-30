# IMPORTS
import pandas as pd

###################################################################################
# GET_SUM FUNCTION

def get_sum(source_file_path, output_file_path, file_name):
  data = pd.read_csv(source_file_path)
  data = data.set_index('Merchant Name')
  dates = pd.Series(pd.period_range(str(data.columns[0]), freq="M", periods=len(data.columns)))
  data.columns = dates
  data_frame = pd.DataFrame({'Month':dates,'total':data.sum(axis=0).values})
  data_frame = data_frame.set_index('Month')
  data_frame.to_csv(output_file_path+file_name+'.csv')
  return data_frame

