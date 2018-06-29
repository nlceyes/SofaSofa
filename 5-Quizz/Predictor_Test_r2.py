# -*- coding: utf-8 -*-
# Prophet预测，用后一年的数据作为验证集，并使用绝对百分比误差均值
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet

# 定义评价函数（绝对百分比误差均值）
def MAPE(y_true, y_pred):
	length = len(y_true)
	if length != len(y_pred):
		print('The numbers of true values and predicted values are different!')
	scores = np.zeros((length, 1))
	for i in range(length):
		scores[i] = abs(y_true[i] - y_pred[i]) / y_true[i]
	return scores.mean()

# 读取数据
df = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")
df_d = df.pop('date')
df_q = df.pop('questions')
df_a = df.pop('answers')

# 训练集与验证集
n_threshold = 1850
train_q = pd.DataFrame()
train_a = pd.DataFrame()
train_q['ds'] = df_d[:n_threshold] # Prophe标准日期格式ds
train_q['y'] = df_q[:n_threshold] # Prophe标准目标值格式y
train_a['ds'] = df_d[:n_threshold] # Prophe标准日期格式ds
train_a['y'] = df_a[:n_threshold] # Prophe标准目标值格式y
y_test_q = df_q[n_threshold:]
y_test_a = df_a[n_threshold:]

# 建立模型并交叉验证
# Parameter Tuning：
# seasonality_mode='multiplicative' for Q
# changepoint_range=0.7 for A and Q
model_q = Prophet(seasonality_mode='multiplicative', changepoint_range=0.7)
model_a = Prophet(changepoint_range=0.7)
for id, model, train, y_test in zip(['Questions', 'Answers'], [model_q, model_a], [train_q, train_a], [y_test_q, y_test_a]):
		print(len(y_test))
		t0 = time.time()
		model.fit(train)
		future = model.make_future_dataframe(periods=403)
		forecast = model.predict(future)
		preds = forecast['yhat'][n_threshold:]
		print(len(preds))
		scores = MAPE(y_test.values, preds.values)
		scores_t = MAPE(train['y'].values, forecast['yhat'][:n_threshold].values)
		print('%s: \t%.6f(%.6f)\tTime: %.1f s' %(id, scores, scores_t, time.time()-t0))