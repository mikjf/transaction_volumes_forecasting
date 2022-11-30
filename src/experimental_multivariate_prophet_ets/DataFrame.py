import json
import pandas as pd
import warnings
from Data_Store_Pred import pred
warnings.filterwarnings('ignore')

with open('data.json', 'r') as fp:
    data_1 = json.load(fp)

df = pd.read_csv("./data/processed/ninety.csv")  ###########THIS MUST BE REPLACED BY A FILE THAT CONTAINS 90% OF TOTAL VOLUME TRANSACTIONS
df = df.set_index('Merchant Name')

df_pred = pred(df, data_1, num_sim=5)
df_pred.to_csv('data/raw/Pred_ETS_5months.csv', encoding='utf-8')








