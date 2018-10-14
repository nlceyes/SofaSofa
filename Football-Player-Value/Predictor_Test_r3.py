# -*- coding: utf-8 -*-
# Category: Goalkeeper, Defender, Offender with More Features
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from datetime import date
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Load Data and Pre-Processing
train = pd.read_csv("train.csv")
# Age
today = date(2018, 3, 5)
train['birth_date'] = pd.to_datetime(train['birth_date'])
train['age'] = (today - train['birth_date']).apply(lambda x: x.days) / 365
# Best Position
positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']
train['best_score'] = train[positions].max(axis=1)
train['best_pos'] = train[positions].idxmax(axis=1)
# Physical(BMI)
train['BMI'] = 10000 * train['weight_kg'] / (train['height_cm'] ** 2)
# train.to_csv('train_rev.csv', index=False)

# Data Preparation
# 'sho', 'pas', 'dri', 'def', 
#, 'league', 'nationality'
cols = ['age', 'height_cm', 'weight_kg', 'BMI', 
		'potential', 'pac', 'sho', 'pas', 'dri', 'def', 'phy', 'international_reputation', 'best_score', 'best_pos']
cols_gk = ['age', 'height_cm', 'weight_kg', 'BMI', 
		'potential', 'pac', 'phy', 'international_reputation', 'best_score']
train_off = train[(train['best_pos']=='cf')|(train['best_pos']=='rw')|(train['best_pos']=='st')|(train['best_pos']=='cam')|(train['best_pos']=='cm')][cols]
train_off_y = train[(train['best_pos']=='cf')|(train['best_pos']=='rw')|(train['best_pos']=='st')|(train['best_pos']=='cam')|(train['best_pos']=='cm')]['y']
# train_off = pd.get_dummies(train_off,columns=['league', 'nationality'])
train_off = pd.get_dummies(train_off,columns=['best_pos'])
train_def = train[(train['best_pos']=='cb')|(train['best_pos']=='rb')|(train['best_pos']=='cdm')][cols]
train_def_y = train[(train['best_pos']=='cb')|(train['best_pos']=='rb')|(train['best_pos']=='cdm')]['y']
# train_def = pd.get_dummies(train_def,columns=['league', 'nationality'])
train_def = pd.get_dummies(train_def,columns=['best_pos'])
train_gk = train[train['best_pos']=='gk'][cols_gk]
train_gk_y = train[train['best_pos']=='gk']['y']
# train_gk = pd.get_dummies(train_gk,columns=['league', 'nationality'])

# Cross-Validation of Baseline Models
# Define Module Function
def Run_Models(model_list, x_train, y_train, flag):
	print('\n%s Part:' %flag)
	for id, clf in model_list:
		t0 = time.time()
		# print(clf)
		scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
		print('%s: \t%.4f\tTime: %.1f s' %(id, scores.mean(), time.time()-t0))
# Run
train_list = (train_gk, train_def, train_off)
train_y_list = (train_gk_y, train_def_y, train_off_y)
category_list = ('Goalkeeper', 'Defender', 'Offender')
model_list = [	('RF', RandomForestRegressor(random_state=0)), 
				('GB', GradientBoostingRegressor(random_state=0)), 
				('XGB', xgb.XGBRegressor()), 
				('LGB', lgb.LGBMRegressor(verbosity=-1)), 
				('CaB', cb.CatBoostRegressor(logging_level='Silent', random_state=0))]
for flag, x_train, y_train in zip(category_list, train_list, train_y_list):
	Run_Models(model_list, x_train, y_train, flag)