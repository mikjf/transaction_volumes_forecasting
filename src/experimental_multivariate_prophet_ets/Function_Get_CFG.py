import pandas as pd
from Modeling_ETS import run_ETS_get_rmse
from datetime import datetime



def store_best_hyperparameter(data, n_test = 5):
  start_time = datetime.now()
  Mod_merch = {}
  data = data.set_index('Merchant Name')
  print('Running...')
  for i in range(len(data.index)):
    df = data.iloc[i]
    start_time_mer = datetime.now()
    cfg, error = run_ETS_get_rmse(df, n_test)
    end_time_mer = datetime.now()
    print(f'{data.index[i]}\nConfig:\t{cfg}\nError(RSME):\t{error}\nTime running: {end_time_mer - start_time_mer}\n######################')
    Mod_merch[data.index[i]] = cfg
  end_time = datetime.now()
  print(f'###Process finished###\nTotal time: {end_time - start_time}')
  return Mod_merch
