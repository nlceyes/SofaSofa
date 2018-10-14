# -*- coding: utf-8 -*-
# Test of One-Hot of Each Feature with LGBM
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# 读取数据
train = pd.read_csv("train.csv")
train.drop('CaseId', axis=1, inplace=True)
y_train = train.pop('Evaluation')
cols = list(train.columns)

# 建立模型并交叉验证
t0 = time.time()
######
# for col in cols:
	# train_onehot = pd.get_dummies(train, columns=[col])
	# clf = lgb.LGBMClassifier()
	# scores = cross_val_score(clf, train_onehot, y_train, cv=5, scoring='average_precision')
	# print('%s: \t%.6f' %(col, scores.mean()))
######
col = ['Q31', 'Q6', 'Q9', 'Q10', 'Q19', 'Q2', 'Q4', 'Q28', 'Q12', 'Q5', 'Q18', 'Q15', 'Q26', 'Q32']
train_onehot = pd.get_dummies(train, columns=col)
clf = lgb.LGBMClassifier()
scores = cross_val_score(clf, train_onehot, y_train, cv=5, scoring='average_precision')
print(scores.mean())
######
print('All Done in %.1f s' % (time.time() - t0))
