import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def pred(df, data_1, num_sim=5):
    data_store = pd.DataFrame()
    for k, v in data_1.items():
        data = df.loc[k]
        t, d, s, s_p, bx, b, sm = v
        model = ExponentialSmoothing(data, trend=t, damped_trend=d, seasonal=s, seasonal_periods=s_p, use_boxcox=bx)
        fit = model.fit(smoothing_level=sm, remove_bias=b)
        predictions = fit.forecast(num_sim)
        data_store[f'{k}'] = list(predictions)
    data_store.index = pd.Series(pd.period_range(start=predictions.index[0], freq="M", periods=num_sim))
    #data_store.index = pd.date_range(start=predictions.index[0], end=predictions.index[-1], freq="M")
    return data_store
