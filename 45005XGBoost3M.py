import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import gc

print('Loading data ...')

#train = pd.read_csv('../input/train_2016.csv')
#prop = pd.read_csv('../input/properties_2016.csv')
#sample = pd.read_csv('../input/sample_submission.csv')

train1 = pd.read_csv("45005h2010.txt", delim_whitespace=True,na_values=[99.00,999.0])
train2 = pd.read_csv("45005h2011.txt", delim_whitespace=True,na_values=[99.00,999.0])
train3 = pd.read_csv("45005h2012.txt", delim_whitespace=True,na_values=[99.00,999.0])
train4 = pd.read_csv("45005h2016.txt", delim_whitespace=True,na_values=[99.00,999.0])

train = pd.concat([train1, train2,train3,train4], ignore_index=True)
#train=train4
del train1, train2, train3, train4; gc.collect()

#train = pd.concat([train1, train3,train4], ignore_index=True)
#del train1, train3, train4; gc.collect()


print('train.shape=',train.shape)
print('Binding to float32')

for c, dtype in zip(train.columns, train.dtypes):
	if dtype == np.float64:
		train[c] = train[c].astype(np.float32)

print('Creating training set ...')

#df_train = train #train.merge(prop, how='left', on='parcelid')
#x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','hashottuborspa','fireplaceflag','taxdelinquencyflag'], axis=1)
#x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'assessmentyear','propertycountylandusecode'],axis=1) # 'storytypeid','pooltypeid2','pooltypeid10','poolcnt','decktypeid', 'buildingclasstypeid'], axis=1)
df_train  = train.drop([ "GST","APD",'DPD', 'MWD','DEWP', 'VIS', 'TIDE','YY','MM','DD','hh','mm'], axis=1)
x_train = df_train.dropna()

y_train = x_train['WVHT'].values  
#x_train  = x_train.drop(['WVHT' ], axis=1)
x_train  = x_train.drop(['WVHT','WDIR','PRES','ATMP','WTMP' ], axis=1)

print('x_train=, y_train=', x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 3000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.9
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 6
params['silent'] = 1
#param['updater'] = 'grow_gpu'
#param['gpu_id'] = 1
#param['max_bin'] = 16
#param['tree_method'] = 'gpu_hist'

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=200, verbose_eval=10)

del d_train, d_valid; gc.collect()


# plot the important features #
#fig, ax = plt.subplots(figsize=(12,18))
#xgb.plot_importance(clf,  height=0.8, ax=ax)
#plt.show()



print('Building test set ...')

#prop = pd.read_csv("45167h2016.txt", delim_whitespace=True,na_values=[99.00,999.0])
#prop = pd.read_csv("45005h2011.txt", delim_whitespace=True,na_values=[99.00,999.0])
#prop = prop.drop([ "GST","APD",'DPD', 'MWD','DEWP', 'VIS', 'TIDE','YY','MM','DD','hh','mm'], axis=1)
#prop = prop.drop(['WDIR','PRES','ATMP','WTMP', "GST","APD",'DPD', 'MWD','DEWP', 'VIS', 'TIDE'], axis=1)
#prop = prop.dropna()


prop = pd.read_csv("new.csv")

print('Writing sample csv ...')
prop.to_csv('sample2012.csv', index=False, float_format='%.4f')  
from shutil import copyfile
copyfile('sample2012.csv','sample2012new.csv')


prop = prop.drop(['WH_diff','WHbuoy','WHmodel','WSP_diff','WSPbuoy'], axis=1)

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
for c in sub.columns[sub.columns != 'WSPD']:
    sub[c] = p_test

print('Writing csv ...')
sub.to_csv('xgb_starter2012.csv', index=False, float_format='%.4f') # Thanks to @inversion

