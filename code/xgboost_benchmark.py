import pandas as pd	
import numpy as np 
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from utility import gini, RI,factorizing,xgb_features,modify_labels,sklearn_features
import pandas as pd
import numpy as np
def make_submission(filename,idx,ypred):
	#generate solution
	preds = pd.DataFrame({"Id": idx, "Hazard": ypred})
	preds = preds.set_index('Id')
	preds.to_csv(filename)
	print ("Storing solution to file %s."%filename)

def get_data_1(train=None,test=None,functions=[np.mean,np.var]):
	#load train and test 
	if train is None:
		train  = pd.read_csv('../input/train.csv', index_col=0)
	if test is None:
		test  = pd.read_csv('../input/test.csv', index_col=0)
	labels = train.Hazard.values
	train.drop('Hazard', axis=1, inplace=True)
	idx = test.index.values
	train,test = factorizing(train,labels,test,functions=functions)
	return train,labels.astype(float),test,idx

def get_data_2(train=None,test=None):
	#load train and test 
	if train is None:
		train  = pd.read_csv('../input/train.csv', index_col=0)
	if test is None:
		test  = pd.read_csv('../input/test.csv', index_col=0)
	labels = train.Hazard.values
	train.drop('Hazard', axis=1, inplace=True)
	idx = test.index.values
	
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
params["eta"] = 0.01
params["min_child_weight"] = 4
params["subsample"] = 0.8
params["colsample_bytree"]=0.8
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 8

train,y,test,idx = get_data_1()
labels = modify_labels(y,padding_function=np.log10)

rtrain,rtest = xgb_features(train,labels,test,params=params,random_state=1,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
meta_features_train = rtrain
meta_features_test = rtest

rtrain,rtest = xgb_features(train,np.log1p(labels),test,params=params,random_state=100,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

params['objective'] = 'reg:logistic'
rtrain,rtest= xgb_features(train,labels,test,params=params,random_state=200,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

rtrain,rtest = xgb_features(train,np.log1p(labels),test,params=params,random_state=300,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

params['objective'] = 'count:poisson'
rtrain,rtest = xgb_features(train,labels,test,params=params,random_state=400,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

rtrain,rtest= xgb_features(train,np.log1p(labels),test,params=params,random_state=500,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

params['objective'] = 'rank:pairwise'
params['eval_metric'] = 'rmse'
rtrain,rtest= xgb_features(train,labels,test,params=params,random_state=600,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

rtrain,rtest= xgb_features(train,np.log1p(labels),test,params=params,random_state=700,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

### SKLEARN ###
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
#RF
rf = RandomForestRegressor(n_estimators=500,n_jobs=-1,max_features=0.4,random_state=499)
rtrain,rtest = sklearn_features(train,labels,test,model=rf,random_state=499,n_folds=5)
for i in range(9):
	rtrain1,rtest1 = sklearn_features(train,labels,test,model=rf,random_state=21+i*3,n_folds=5)
	rtrain+=rtrain1
	rtest += rtest1

rtrain/=10
rtest/=10
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

#et
et = ExtraTreesRegressor(max_features=0.4,n_jobs=-1,n_estimators=300,random_state=12)
rtrain,rtest = sklearn_features(train,labels,test,model=et,random_state=119,n_folds=5)
for i in range(9):
	rtrain1,rtest1 = sklearn_features(train,labels,test,model=et,random_state=99+i*3,n_folds=5)
	rtrain+=rtrain1
	rtest +=rtest1

rtrain/=10
rtest/=10
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))
#
ada = AdaBoostRegressor()
rtrain,rtest = sklearn_features(train,labels,test,model=ada,random_state=22,n_folds=5)
for i in range(9):
	rtrain1,rtest1 = sklearn_features(train,labels,test,model=ada,random_state=199+i*3,n_folds=5)
	rtrain+=rtrain1
	rtest +=rtest1

rtrain/=10
rtest/=10
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))
### SVR ###
from sklearn.svm import SVR,NuSVR
scl = StandardScaler().fit(train)
trainn = scl.transform(train)
testn = scl.transform(test)
#SVR
svr1 = SVR(C=2.,tol=1e-5)
rtrain,rtest = sklearn_features(trainn,labels,testn,model=svr1,random_state=9991,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

rtrain,rtest = sklearn_features(trainn,np.log(labels),testn,model=svr1,random_state=91,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))
#nuSVR
nuSvr = NuSVR(C=10.)
rtrain,rtest = sklearn_features(trainn,labels,testn,model=nuSvr,random_state=666,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

rtrain,rtest = sklearn_features(trainn,np.log(labels),testn,model=nuSvr,random_state=555,n_folds=5)
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

### LINEAR_MODEL ###
from sklearn.linear_model import SGDRegressor,TheilSenRegressor
##
sgdr = SGDRegressor(loss='epsilon_insensitive',penalty='elasticnet',random_state=11,eta0=0.001,alpha=0.00001,n_iter=50,l1_ratio=0.1,epsilon=0.4,average =100,power_t=.1)
rtrain,rtest = sklearn_features(trainn,y,testn,model=sgdr,random_state=2211,n_folds=5)
for i in range(9):
	rtrain1,rtest1 = sklearn_features(train,y,test,model=sgdr,random_state=129+i*3,n_folds=5)
	rtrain+=rtrain1
	rtest +=rtest1

rtrain/=10
rtest/=10
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

##TheilSenRegressor
th = TheilSenRegressor(n_jobs=-1)
rtrain,rtest = sklearn_features(trainn,y,testn,model=th,random_state=1662,n_folds=5)
for i in range(9):
	rtrain1,rtest1 = sklearn_features(train,y,test,model=th,random_state=1919+i*3,n_folds=5)
	rtrain+=rtrain1
	rtest +=rtest1

rtrain/=10
rtest/=10
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

from sklearn.linear_model import RandomizedLogisticRegression
rlr = RandomizedLogisticRegression(n_jobs=-1,random_state=1)
rtrain,rtest = sklearn_features(trainn,y,testn,model=rlr,random_state=77,n_folds=5)
for i in range(9):
	rtrain1,rtest1 = sklearn_features(train,y,test,model=rlr,random_state=102+i*11,n_folds=5)
	rtrain+=rtrain1
	rtest +=rtest1

rtrain/=10
rtest/=10
print 'gini score is %f '%gini(y,rtrain)
np.column_stack((meta_features_train,rtrain))
np.column_stack((meta_features_test,rtest))

mt = pd.DataFrame(data=meta_features_train)
mt.to_csv('meta_features_train.csv',index=False)

mt = pd.DataFrame(data=meta_features_test)
mt.to_csv('meta_features_test.csv',index=False)

#params = {}
#params["objective"] = 'multi:softmax' #'reg:logistic'#"reg:linear" #
#params["eta"] = 0.03
#params["min_child_weight"] = 4
#params["subsample"] = 0.8
#params["colsample_bytree"]=0.8
#params["scale_pos_weight"] = 1.0
#params["silent"] = 1
#params["max_depth"] = 8
#params["num_class"] = 5
#params['eval_metric'] = 'mlogloss'
#yc = y.__copy__()
#yc[yc>4] = 5
#yc = yc-1
#rtrain9,rtest9 = xgb_features(train,yc,test,params=params,random_state=800,n_folds=5)
##params["eval_metric"]='auc'