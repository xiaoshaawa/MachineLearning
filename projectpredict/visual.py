import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

def calculate_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = data['收盘'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['收盘'].ewm(span=slow_period, adjust=False).mean()
    data['MACD'] = fast_ema - slow_ema
    data['信号线'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
    data['柱状图'] = data['MACD'] - data['信号线']
    return data

def visualize_data(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # K线图
    fig, ax = plt.subplots(figsize=(20,7))
    for i, row in data.iterrows():
        color = 'green' if row['收盘'] >= row['开盘'] else 'red'
        ax.plot([row['日期'], row['日期']], [row['低'], row['高']], color=color)
        ax.plot([row['日期'], row['日期']], [row['开盘'], row['收盘']], linewidth=6, color=color)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=0)
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.title('K线图')
    plt.grid(True)
    plt.show()
    # 收盘价趋势
    plt.figure(figsize=(20, 6))
    plt.plot(data['日期'], data['收盘'], label='收盘价')
    plt.xlabel('日期')
    plt.ylabel('收盘价')
    plt.title('收盘价趋势')
    plt.xticks(rotation=0)
    plt.legend()
    plt.grid(True)
    plt.show()
    # 移动平均线
    data['移动平均线_5天'] = data['收盘'].rolling(window=5).mean()
    data['移动平均线_20天'] = data['收盘'].rolling(window=20).mean()
    plt.figure(figsize=(20, 6))
    plt.plot(data['日期'], data['收盘'], label='收盘价')
    plt.plot(data['日期'], data['移动平均线_5天'], label='5天移动平均线')
    plt.plot(data['日期'], data['移动平均线_20天'], label='20天移动平均线')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.title('移动平均线分析')
    plt.xticks(rotation=0)
    plt.legend()
    plt.grid(True)
    plt.show()
    # RSI
    data['RSI'] = calculate_rsi(data['收盘'])
    plt.figure(figsize=(20, 6))
    plt.plot(data['日期'], data['RSI'], label='RSI')
    plt.xlabel('日期')
    plt.ylabel('RSI')
    plt.title('相对强弱指数（RSI）')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.legend()
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()
    # 布林带
    data['标准差'] = data['收盘'].rolling(window=20).std()
    data['上轨'] = data['移动平均线_20天'] + (data['标准差'] * 2)
    data['下轨'] = data['移动平均线_20天'] - (data['标准差'] * 2)
    plt.figure(figsize=(20, 6))
    plt.plot(data['日期'], data['收盘'], label='收盘价')
    plt.plot(data['日期'], data['上轨'], label='上轨', linestyle='--')
    plt.plot(data['日期'], data['下轨'], label='下轨', linestyle='--')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.title('布林带')
    plt.legend()
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()
    # MACD
    data = calculate_macd(data)
    plt.figure(figsize=(20, 6))
    plt.plot(data['日期'], data['MACD'], label='MACD')
    plt.plot(data['日期'], data['信号线'], label='信号线', linestyle='--')
    plt.bar(data['日期'], data['柱状图'], label='柱状图', color='grey', alpha=0.5)
    plt.xlabel('日期')
    plt.ylabel('指标值')
    plt.title('MACD 指标')
    plt.legend()
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()
    return data 