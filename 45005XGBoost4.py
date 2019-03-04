import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import gc

print('Loading data ...')


train = pd.read_csv("new.csv")

print('train.shape=',train.shape)
print('Binding to float32')

for c, dtype in zip(train.columns, train.dtypes):
	if dtype == np.float64:
		train[c] = train[c].astype(np.float32)

print('Creating training set ...')

df_train  = train.drop([ "WSPbuoy","WSP_diff",'WHmodel', 'WHbuoy'], axis=1)
#df_train  = train.drop([ 'WHmodel', 'WHbuoy'], axis=1)
x_train = df_train.dropna()


y_train = x_train['WH_diff'].values  
x_train  = x_train.drop(['WH_diff' ], axis=1)
prop = x_train


print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 4000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
"""
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1
"""
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid; gc.collect()


# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(clf,  height=0.8, ax=ax)
plt.show()


print('Building test set ...')

print('Writing sample csv ...')
prop.to_csv('sample2012.csv', index=False, float_format='%.4f')  

df_test = prop 

del prop; gc.collect()

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

#print(x_test[1000:1010])

del df_test; gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test; gc.collect()

sub = pd.read_csv('sample2012.csv')
for c in sub.columns[sub.columns != 'WVHT']:
    sub[c] = p_test

print('Writing csv ...')
sub.to_csv('xgb_starter2012err.csv', index=False, float_format='%.4f') # Thanks to @inversion
