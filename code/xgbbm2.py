#
import pandas as pd	
import numpy as np 
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from utility import gini, RI,factorizing,xgb_features,modify_labels,sklearn_features
import pandas as pd
import numpy as np
def get_data_1(train=None,test=None,functions=[np.mean,np.var],dropping=False):
	#load train and test 
	if train is None:
		train  = pd.read_csv('../input/train.csv', index_col=0)
	if test is None:
		test  = pd.read_csv('../input/test.csv', index_col=0)
	labels = train.Hazard.values
	train = train.drop('Hazard', axis=1)
	idx = test.index.values
	if dropping:
		train = train.drop('T2_V10', axis=1)
		train = train.drop('T2_V7', axis=1)
		train = train.drop('T1_V13', axis=1)
		train = train.drop('T1_V10', axis=1)
		test = test.drop('T2_V10', axis=1)
		test = test.drop('T2_V7', axis=1)
		test = test.drop('T1_V13', axis=1)
		test = test.drop('T1_V10', axis=1)
	train,test = factorizing(train,labels,test,functions=functions)
	return train,labels.astype(float),test,idx

def get_data_2(train=None,test=None,dropping=False):
	#load train and test 
	if train is None:
		train  = pd.read_csv('../input/train.csv', index_col=0)
	if test is None:
		test  = pd.read_csv('../input/test.csv', index_col=0)
	labels = train.Hazard.values
	train = train.drop('Hazard', axis=1)
	idx = test.index.values
	if dropping:
		train = train.drop('T2_V10', axis=1)
		train = train.drop('T2_V7', axis=1)
		train = train.drop('T1_V13', axis=1)
		train = train.drop('T1_V10', axis=1)
		test = test.drop('T2_V10', axis=1)
		test = test.drop('T2_V7', axis=1)
		test = test.drop('T1_V13', axis=1)
		test = test.drop('T1_V10', axis=1)
	from sklearn.feature_extraction import DictVectorizer
	train = train.T.to_dict().values()
	test = test.T.to_dict().values()
	vec = DictVectorizer()
	train = vec.fit_transform(train)
	test = vec.transform(test)	
	return train,labels.astype(float),test,idx

### XGBOOST ###	
params = {}
params["objective"] = 'reg:linear'
params["eta"] = 0.007
params["min_child_weight"] = 5
params["subsample"] = 0.8
params["colsample_bytree"]=0.8
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 8

train,y,test,idx = get_data_1(dropping=True)
labels = modify_labels(y,padding_function=np.log10)

rtrain,rtest = xgb_features(train,labels,test,params=params,random_state=1,n_folds=10,early_stop=100)
print 'gini score is %f '%gini(y,rtrain)
meta_features_train = rtrain
meta_features_test = rtest

rtrain,rtest = xgb_features(train,np.log1p(labels),test,params=params,random_state=100,n_folds=10,early_stop=100)
print 'gini score is %f '%gini(y,rtrain)
meta_features_train= np.column_stack((meta_features_train,rtrain))
meta_features_test = np.column_stack((meta_features_test,rtest))

params['objective'] = 'reg:logistic'
rtrain,rtest= xgb_features(train,labels,test,params=params,random_state=200,n_folds=10,early_stop=100)
print 'gini score is %f '%gini(y,rtrain)
meta_features_train= np.column_stack((meta_features_train,rtrain))
meta_features_test = np.column_stack((meta_features_test,rtest))

rtrain,rtest = xgb_features(train,np.log1p(labels),test,params=params,random_state=300,n_folds=10,early_stop=100)
print 'gini score is %f '%gini(y,rtrain)
meta_features_train= np.column_stack((meta_features_train,rtrain))
meta_features_test = np.column_stack((meta_features_test,rtest))

params['objective'] = 'count:poisson'
rtrain,rtest = xgb_features(train,labels,test,params=params,random_state=400,n_folds=10,early_stop=100)
print 'gini score is %f '%gini(y,rtrain)
meta_features_train= np.column_stack((meta_features_train,rtrain))
meta_features_test = np.column_stack((meta_features_test,rtest))

rtrain,rtest= xgb_features(train,np.log1p(labels),test,params=params,random_state=500,n_folds=10,early_stop=100)
print 'gini score is %f '%gini(y,rtrain)
meta_features_train= np.column_stack((meta_features_train,rtrain))
meta_features_test = np.column_stack((meta_features_test,rtest))

params['objective'] = 'rank:pairwise'
params['eval_metric'] = 'rmse'
rtrain,rtest= xgb_features(train,labels,test,params=params,random_state=600,n_folds=10,early_stop=100)
print 'gini score is %f '%gini(y,rtrain)
meta_features_train= np.column_stack((meta_features_train,rtrain))
meta_features_test = np.column_stack((meta_features_test,rtest))

rtrain,rtest= xgb_features(train,np.log1p(labels),test,params=params,random_state=700,n_folds=10,early_stop=100)
print 'gini score is %f '%gini(y,rtrain)
meta_features_train= np.column_stack((meta_features_train,rtrain))
meta_features_test = np.column_stack((meta_features_test,rtest))

mt = pd.DataFrame(data=meta_features_train)
mt.to_csv('meta_features_train_2.csv',index=False)

mt = pd.DataFrame(data=meta_features_test)
mt.to_csv('meta_features_test_2.csv',index=False)