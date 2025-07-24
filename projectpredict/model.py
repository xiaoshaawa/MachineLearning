from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(look_back, units=50, dropout=None, optimizer='adam', lstm_units_list=None):
    # 支持多层LSTM，每层单元数可不同
    layers = []
    if lstm_units_list is not None and len(lstm_units_list) > 0:
        for i, u in enumerate(lstm_units_list):
            if i == 0:
                layers.append(LSTM(u, return_sequences=(len(lstm_units_list) > 1), input_shape=(look_back, 1)))
            elif i < len(lstm_units_list) - 1:
                layers.append(LSTM(u, return_sequences=True))
            else:
                layers.append(LSTM(u, return_sequences=False))
            if dropout is not None:
                layers.append(Dropout(dropout))
    else:
        layers.append(LSTM(units, return_sequences=False, input_shape=(look_back, 1)))
        if dropout is not None:
            layers.append(Dropout(dropout))
    layers.append(Dense(1))
    model = Sequential(layers)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"均方误差（MSE）: {mse}")
    print(f"均方根误差（RMSE）: {rmse}")
    print(f"平均绝对误差（MAE）: {mae}")
    return mse, rmse, mae

def hyperparameter_search(x, y, look_back, param_distribs):
    best_score = float('inf')
    best_params = None
    for units, dropout, optimizer, batch_size, epochs in itertools.product(
            param_distribs['units'], 
            param_distribs['dropout'], 
            param_distribs['optimizer'], 
            param_distribs['batch_size'], 
            param_distribs['epochs']):
        model = build_lstm_model(look_back, units=units, dropout=dropout, optimizer=optimizer)
        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[early_stopping])
        y_pred = model.predict(x)
        score = mean_squared_error(y, y_pred)
        if score < best_score:
            best_score = score
            best_params = {
                'units': units,
                'dropout': dropout,
                'optimizer': optimizer,
                'batch_size': batch_size,
                'epochs': epochs
            }
    return best_params, best_score

def predict_and_plot(model, x, y, scaler, label, actual_label):
    predictions = model.predict(x)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y.reshape(-1, 1))
    evaluate_model(actual, predictions)
    plt.figure(figsize=(20,6))
    plt.plot(actual, label=actual_label)
    plt.plot(predictions, label=label)
    plt.legend()
    plt.xlabel('样本编号')
    plt.ylabel('收盘价')
    plt.title(label)
    plt.show()
    return predictions, actual 