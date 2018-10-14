# -*- coding: utf-8 -*-
# Use Important One-Hot Features and other models
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score

# 读取数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit = pd.read_csv("sample_submit.csv")
train.drop('CaseId', axis=1, inplace=True)
test.drop('CaseId', axis=1, inplace=True)
y_train = train.pop('Evaluation')

# One-Hot of Important Features
# col = ['Q31', 'Q6', 'Q9', 'Q10', 'Q19', 'Q2', 'Q4', 'Q28', 'Q12', 'Q5', 'Q18', 'Q15', 'Q26', 'Q32']
# train_onehot = pd.get_dummies(train, columns=col)
# test_onehot = pd.get_dummies(test, columns=col)
train_onehot = train
test_onehot = test

# 建模并预测
clf = RandomForestClassifier(n_estimators=200, min_samples_split=32, random_state=0)
scores = cross_val_score(clf, train_onehot, y_train, cv=5, scoring='average_precision')
print(scores.mean())
clf.fit(train_onehot, y_train)
y_train_pred = clf.predict_proba(train_onehot)[:, 1]
score = average_precision_score(y_train, y_train_pred)
print(score)
y_test_pred = clf.predict_proba(test_onehot)[:, 1]

# 输出结果
submit['Evaluation'] = y_test_pred
submit.to_csv('RF_pred_onehot.csv', index=False)
