from load_and_output_top_merchants import select_merchants
from get_sum import get_sum
import pandas as pd
###################################################################################
# FILE PATHS
original_file_path = 'data/raw/Mock_Time_Series_Merchants_Transactions_Anonymized.csv'
source = 'data/raw/Mock_Time_Series_Merchants_Transactions_Anonymized_dates_changed.csv'
output = 'data/raw/Mock_'
###################################################################################
# FUNCTION TO DEAL WITH DATE FORMAT
def sortDates(datesList):
   split_up = datesList.split('-')
   return split_up[1], split_up[0]

def getting_dates(df):
  df_trans_set = df.set_index('Merchant Name')
  months = {'Jan':'01','Feb':'02','Mär':'03','Apr':'04', 'Mai':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Okt':'10', 'Nov':'11', 'Dez':'12'}
  store = list(df_trans_set.columns)
  for i, n in enumerate(store):
    for g, k in months.items():
      if g in n:
        store[i] = store[i].replace(f'{g} ', f'{k}-')
        store[i] = store[i].replace('-', '-20')
  store.sort(key=sortDates)
  star_date = store[0]
  last_date = store[-1]
  return star_date, last_date, store
###################################################################################
# REPLACING THE COLUMN NAME, SO THE DATES ARE EASIER TO WORK WITH
# COMMENT OUT IF NOT NEEDED
data = pd.read_csv(original_file_path)
start_date, last_date, store = getting_dates(data)
store.insert(0,'Merchant Name')
data.columns = store
data = data.set_index('Merchant Name')
data.to_csv(source)
###################################################################################
# SELECTS AND OUTPUTS THE SELECTED MERCHANTS
_, selected_file_name = select_merchants(source_file_path = source, output_file_path = output, top=True, percentage=0.75)
###################################################################################
# SUMS UP GROUP OF MERCHANTS AND OUTPUTS THE FILE
get_sum(source_file_path = output+selected_file_name, output_file_path = output, file_name = 'totals_'+selected_file_name)
