

def load_stock_sh000001_data():
    import os
    import pandas as pd
    data_all = pd.read_csv(os.path.join(os.path.dirname(__file__), 'sh000001.csv'))
    data_x = data_all[['open', 'close', 'low', 'high', 'volume', 'money', 'change']].to_numpy()
    data_y = data_all[['label']].to_numpy()
    return data_x, data_y
