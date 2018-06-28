# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
from collections import Counter
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# 读取数据
train = pd.read_table('train.txt', ',')
test = pd.read_table('test.txt', ',')
submit = pd.read_csv('sample_submit.csv')

# 所有男生的名字
train_male = train[train['gender'] == 1]
m_cnt = len(train_male)
names_male = "".join(train_male['name'])

# 所有女生的名字
train_female = train[train['gender'] == 0]
f_cnt = len(train_female)
names_female = "".join(train_female['name'])

# 统计每个字在男生、女生名字中出现的总次数
lists_male = names_male
counts_male = Counter(lists_male)
lists_female = names_female
counts_female = Counter(lists_female)

# 得到训练集中每个人的每个字的词频（Term Frequency，通常简称TF）
train_encoded = []
for i in range(len(train)):
    name = train.at[i, 'name']
    chs = name
    row = [0., 0., 0., 0, train.at[i, 'gender']]
    for j in range(len(chs)):
        row[2* j] = counts_female[chs[j]] * 1. / f_cnt
        row[2* j + 1] = counts_male[chs[j]] * 1. / m_cnt
    train_encoded.append(row)

# 得到测试集中每个人的每个字的词频（Term Frequency，通常简称TF）
test_encoded = []
for i in range(len(test)):
    name = test.at[i, 'name']
    chs = name
    row = [0., 0., 0., 0.,]
    for j in range(len(chs)):
        try:
            row[2 * j] = counts_female[chs[j]] * 1. / f_cnt
        except:
            pass
        try:
            row[2 * j + 1] = counts_male[chs[j]] * 1. / m_cnt
        except:
            pass
    test_encoded.append(row)

# 转换为pandas.DataFrame的形式
# 1_f是指这个人的第一个字在训练集中所有女生的字中出现的频率
# 2_f是指这个人的第二个字在训练集中所有女生的字中出现的频率
# 1_m是指这个人的第一个字在训练集中所有男生的字中出现的频率
# 2_m是指这个人的第二个字在训练集中所有男生的字中出现的频率
train_encoded = pd.DataFrame(train_encoded, columns=['1_f', '1_m', '2_f', '2_m', 'gender'])
test_encoded = pd.DataFrame(test_encoded, columns=['1_f', '1_m', '2_f', '2_m'])
y_train_encoded = train_encoded.pop('gender')

# 建立模型并交叉验证
t0 = time.time()
model = lgb.LGBMClassifier()
params = {	'boosting_type': ['dart'],
			'num_leaves': [23, 31, 39],
			'learning_rate': [0.25, 0.2, 0.15],
			'n_estimators': [200, 300, 400],
			'min_split_gain': [0, 0.1, 0.2],
			'min_child_samples': [25, 50, 75],
			'subsample': [0.75, 1.0]}
clf = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs = 1)
clf.fit(train_encoded, y_train_encoded)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
means_t = clf.cv_results_['mean_train_score']
stds_t = clf.cv_results_['std_train_score']
param_list = clf.cv_results_['params']
for mean, std, mean_t, std_t, param in zip(means, stds, means_t, stds_t, param_list):
	print("%0.5f (+/-%0.5f) %0.5f (+/-%0.5f) %r" % (mean_t, std_t, mean, std, param))
print(clf.best_params_)
print(clf.best_score_)
print('All Done in %.3f s' % (time.time() - t0))
