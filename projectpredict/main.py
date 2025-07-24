import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import warnings
from tensorflow.keras.callbacks import EarlyStopping

register_matplotlib_converters()
warnings.filterwarnings('ignore')

from data import preprocess_data, feature_engineering
from visual import visualize_data
from model import build_lstm_model, evaluate_model, hyperparameter_search, predict_and_plot

def main():
    data = preprocess_data("../data.csv")
    print('数据维度:', data.shape)
    print('数据信息:')
    print(data.info())
    print('重复值数量:', data.duplicated().sum())
    print(data.head())

    data = visualize_data(data)
    x, y, scaler, look_back = feature_engineering(data)
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model = build_lstm_model(look_back)
    model.fit(x, y, epochs=20, batch_size=32, verbose=1, callbacks=[early_stopping])
    print("\n原始模型评估：")
    predictions, actual = predict_and_plot(model, x, y, scaler, '预测值', '真实值')

    param_distribs = {
        'units': [10, 20, 50, 100],
        'dropout': [0.2, 0.3, 0.4],
        'optimizer': ['adam', 'rmsprop'],
        'batch_size': [16, 32, 64],
        'epochs': [20, 30, 50]
    }
    best_params, best_score = hyperparameter_search(x, y, look_back, param_distribs)
    print("最佳参数:", best_params)
    print("最佳模型得分:", best_score)
    optimized_model = build_lstm_model(
        look_back,
        units=best_params['units'],
        dropout=best_params['dropout'],
        optimizer=best_params['optimizer'],
        lstm_units_list=[50, 30, 10]
    )
    early_stopping_opt = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    optimized_model.fit(x, y, epochs=best_params['epochs'], batch_size=best_params['batch_size'], callbacks=[early_stopping_opt])
    print("\n优化后模型评估：")
    predictions_optimized, _ = predict_and_plot(
        optimized_model, x, y, scaler, '优化参数后的模型预测值', '真实值')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,6))
    plt.plot(actual, label='真实值')
    plt.plot(predictions, label='优化参数前的模型预测值')
    plt.plot(predictions_optimized, label='优化参数后的模型预测值')
    plt.legend()
    plt.xlabel('样本编号')
    plt.ylabel('收盘价')
    plt.title('优化参数前后的 LSTM 预测结果对比')
    plt.show()

if __name__ == "__main__":
    main() 