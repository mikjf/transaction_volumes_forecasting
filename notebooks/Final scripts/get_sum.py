import pandas as pd

def get_sum(start_month, data, file_name):
  data = data.set_index('Merchant Name')
  dates = pd.Series(pd.period_range(start_month, freq="M", periods=len(data.columns)))
  data.columns = dates
  #data.sum(axis=0).to_csv(file_name+'.csv', header=False)
  data_get_sum = pd.DataFrame({'Month': dates, 'total': data.sum(axis=0).values})
  return data_get_sum