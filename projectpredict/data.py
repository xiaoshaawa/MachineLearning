import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    data['日期'] = pd.to_datetime(data['日期'])
    def parse_volume(volume):
        if isinstance(volume, str):
            if 'M' in volume:
                return float(volume.replace('M', '')) * 1_000_000
            elif 'K' in volume:
                return float(volume.replace('K', '')) * 1_000
        return float(volume)
    data['交易量'] = data['交易量'].apply(parse_volume)
    data = data.sort_values(by='日期').reset_index(drop=True)
    return data

def create_dataset(data, look_back=1):
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)

def feature_engineering(data, look_back=5):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data[['收盘']])
    x, y = create_dataset(data_scaled, look_back=look_back)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler, look_back 