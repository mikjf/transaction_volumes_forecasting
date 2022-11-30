import pandas as pd
from Function_Get_CFG import store_best_hyperparameter
import json

df = pd.read_csv("./data/processed/ninety.csv")  ###########THIS MUST BE REPLACED BY A FILE THAT CONTAINS 90% OF TOTAL VOLUME TRANSACTIONS

data = store_best_hyperparameter(df)

with open('data.json', 'w') as fp:
    json.dump(data, fp)
