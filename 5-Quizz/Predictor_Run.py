# -*- coding: utf-8 -*-
# Prophet预测
import time
import numpy as np
import pandas as pd
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
submit = pd.read_csv("sample_submit.csv")
df_d = df.pop('date')
df_q = df.pop('questions')
df_a = df.pop('answers')

# 训练集
train_q = pd.DataFrame()
train_a = pd.DataFrame()
train_q['ds'] = df_d # Prophe标准日期格式ds
train_q['y'] = df_q # Prophe标准目标值格式y
train_a['ds'] = df_d # Prophe标准日期格式ds
train_a['y'] = df_a # Prophe标准目标值格式y

# 建立模型并交叉验证
# Parameter Tuning：
# seasonality_mode='multiplicative' for Q
# changepoint_range=0.7 for A and Q
model_q = Prophet(seasonality_mode='multiplicative', changepoint_range=0.7)
model_a = Prophet(changepoint_range=0.7)
for id, model, train in zip(['questions', 'answers'], [model_q, model_a], [train_q, train_a]):
		t0 = time.time()
		model.fit(train)
		future = model.make_future_dataframe(periods=152)
		forecast = model.predict(future)
		preds = forecast['yhat'][2253:]
		print(len(preds))
		submit[id] = preds.values
		scores = MAPE(train['y'].values, forecast['yhat'][:2253].values)
		print('%s: \t%.6f\tTime: %.1f s' %(id, scores, time.time()-t0))
		model.plot(forecast)
		plt.show()
submit.to_csv('Prophet_Pred.csv', index=False)