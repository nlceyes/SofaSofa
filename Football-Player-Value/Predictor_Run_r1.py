# -*- coding: utf-8 -*-
# Category: Goalkeeper, Defender, Offender with More Features
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from datetime import date
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Load Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit = pd.read_csv("sample_submit.csv")

# Data Pre-Processing
# Age
today = date(2018, 3, 5)
train['birth_date'] = pd.to_datetime(train['birth_date'])
train['age'] = (today - train['birth_date']).apply(lambda x: x.days) / 365
test['birth_date'] = pd.to_datetime(test['birth_date'])
test['age'] = (today - test['birth_date']).apply(lambda x: x.days) / 365
# Best Position
positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']
train['best_score'] = train[positions].max(axis=1)
train['best_pos'] = train[positions].idxmax(axis=1)
test['best_score'] = test[positions].max(axis=1)
test['best_pos'] = test[positions].idxmax(axis=1)
# Physical(BMI)
train['BMI'] = 10000 * train['weight_kg'] / (train['height_cm'] ** 2)
test['BMI'] = 10000 * test['weight_kg'] / (test['height_cm'] ** 2)
# test.to_csv('test_rev.csv', index=False)

# Cross-Validation and Run
test['pred'] = 0
cols = ['height_cm', 'weight_kg', 'BMI', 'potential', 'pac',
        'phy', 'international_reputation', 'age', 'best_score']
cols_off = ['age', 'height_cm', 'weight_kg', 'BMI', 
			'potential', 'pac', 'sho', 'pas', 'dri', 'def', 'phy', 'international_reputation', 'best_score', 'best_pos']
# Defender
clf_def = RandomForestRegressor(criterion='mse', n_estimators=140, min_samples_split=2, random_state=6)
train_def = train[(train['best_pos']=='cb')|(train['best_pos']=='rb')|(train['best_pos']=='cdm')][cols]
train_def_y = train[(train['best_pos']=='cb')|(train['best_pos']=='rb')|(train['best_pos']=='cdm')]['y']
test_def = test[(test['best_pos']=='cb')|(test['best_pos']=='rb')|(test['best_pos']=='cdm')][cols]
scores_def = cross_val_score(clf_def, train_def, train_def_y, cv=5, scoring='neg_mean_absolute_error')
print('Defense Players:')
print(scores_def)
print(scores_def.mean())
clf_def.fit(train_def, train_def_y)
preds_def = clf_def.predict(test_def)
test.loc[(test['best_pos']=='cb')|(test['best_pos']=='rb')|(test['best_pos']=='cdm'), 'pred'] = preds_def
# Offender
# clf_off = lgb.LGBMRegressor(boosting_type='dart', learning_rate=0.25, min_child_samples=5, min_split_gain=0.2, n_estimators=400, num_leaves=63, subsample=0.7)
clf_off = lgb.LGBMRegressor(boosting_type='dart', learning_rate=0.2, min_child_samples=5, min_split_gain=0.0, n_estimators=500, num_leaves=31, subsample=0.8)
# clf_off = lgb.LGBMRegressor(boosting_type='dart', learning_rate=0.3, min_child_samples=4, min_split_gain=0.0, n_estimators=500, num_leaves=31, reg_alpha=1, reg_lambda=10, subsample=0.8)
# clf_off = lgb.LGBMRegressor(boosting_type='dart', learning_rate=0.3, min_child_samples=1, min_split_gain=0.1, n_estimators=600, num_leaves=39, subsample=0.9)
train_off = train[(train['best_pos']=='cm')|(train['best_pos']=='cam')|(train['best_pos']=='cf')|(train['best_pos']=='rw')|(train['best_pos']=='st')][cols_off]
train_off_y = train[(train['best_pos']=='cm')|(train['best_pos']=='cam')|(train['best_pos']=='cf')|(train['best_pos']=='rw')|(train['best_pos']=='st')]['y']
test_off = test[(test['best_pos']=='cm')|(test['best_pos']=='cam')|(test['best_pos']=='cf')|(test['best_pos']=='rw')|(test['best_pos']=='st')][cols_off]
train_off = pd.get_dummies(train_off,columns=['best_pos'])
test_off = pd.get_dummies(test_off,columns=['best_pos'])
scores_off = cross_val_score(clf_off, train_off, train_off_y, cv=5, scoring='neg_mean_absolute_error')
print('Offense Players:')
print(scores_off)
print(scores_off.mean())
clf_off.fit(train_off, train_off_y)
preds_off = clf_off.predict(test_off)
test.loc[(test['best_pos']=='cm')|(test['best_pos']=='cam')|(test['best_pos']=='cf')|(test['best_pos']=='rw')|(test['best_pos']=='st'), 'pred'] = preds_off
# Goalkeeper
# Other than Rev.13 and 23
# clf_gk = GradientBoostingRegressor(learning_rate=0.04, max_depth=6, n_estimators=300, subsample=0.7, random_state=2)
# Rev.13 and 23
clf_gk = xgb.XGBRegressor(booster='dart', gamma=0.0, learning_rate=0.01, max_depth=11, n_estimators=500, subsample=0.6, reg_alpha=0, reg_lambda=1)
train_gk = train[train['best_pos']=='gk'][cols]
train_gk_y = train[train['best_pos']=='gk']['y']
test_gk = test[test['best_pos']=='gk'][cols]
scores_gk = cross_val_score(clf_gk, train_gk, train_gk_y, cv=5, scoring='neg_mean_absolute_error')
print('Goal Keeper:')
print(scores_gk)
print(scores_gk.mean())
clf_gk.fit(train_gk, train_gk_y)
preds_gk = clf_gk.predict(test_gk)
test.loc[test['best_pos']=='gk', 'pred'] = preds_gk

# 输出预测结果至my_XGB_prediction.csv
submit['y'] = test['pred']
submit.to_csv('my_prediction.csv', index=False)