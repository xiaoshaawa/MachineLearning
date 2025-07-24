# 股票价格分析与预测项目

本项目基于LSTM神经网络和常用技术指标（K线、均线、RSI、布林带、MACD等）对股票价格进行分析与预测。

## 目录结构

```
projectpredict/
  ├── data.py
  ├── main.py
  ├── model.py
  ├── README.md
  ├── requirements.txt
  └── visual.py
```

## 依赖安装

请先确保已安装Python 3.7及以上版本。

在`projectpredict`目录下，使用如下命令安装依赖：

```
pip install -r requirements.txt
```

## 数据准备

请将原始数据文件`data.csv`放在`projectpredict'`目录的上一级（即与`projectpredict`文件夹同级）。

## 运行方法
在`projectpredict`目录下，运行：
python main.py
程序将自动读取数据，进行数据分析、可视化和LSTM预测。
