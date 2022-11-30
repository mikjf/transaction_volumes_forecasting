import pandas as pd

def get_top(data, top_percent = 0.9):
  data_ordered = data.copy(deep=True)
  data_ordered['total'] = data.sum(axis='columns')
  data_ordered = data_ordered.sort_values(by='total', ascending = False).copy(deep=True)
  data_ordered['running_total']=data_ordered['total'].cumsum()
  data_ordered['running/total']=data_ordered['running_total']/data_ordered['total'].sum()
  top_merchants = data_ordered[data_ordered['running/total']<=top_percent].drop(columns=['total','running_total','running/total'])
  return top_merchants

def get_bottom(data, bottom_percent = 0.1):
  data_ordered = data.copy(deep=True)
  data_ordered['total'] = data.sum(axis='columns')
  data_ordered = data_ordered.sort_values(by='total', ascending = True).copy(deep=True)
  data_ordered['running_total']=data_ordered['total'].cumsum()
  data_ordered['running/total']=data_ordered['running_total']/data_ordered['total'].sum()
  bottom_merchants = data_ordered[data_ordered['running/total']<=bottom_percent].drop(columns=['total','running_total','running/total'])
  return bottom_merchants

def select_merchants(source_file_path, output_file_path, start_month, top=True, percentage=0.9):
# top = True - selects the merchants with the highest number of transations
# top = False - selects the merchants with the lowest number of transactions
# percentage = what should the ratio of total of the selected merchants to the total volume of transations be
# default setting outputs dataframe with merchants with highest total number of transactions
# that bring 90 % of all transactions
  data = pd.read_csv(source_file_path)
  data = data.set_index('Merchant Name')
  dates = pd.Series(pd.period_range(start_month, freq="M", periods=len(data.columns)))
  data.columns = dates
  if top:
    get_top(data,percentage).to_csv(output_path+'top_'+str(percentage)+'.csv')
  else:
    get_bottom(data,percentage).to_csv(output_path+'bottom_'+str(percentage)+'.csv')
  return