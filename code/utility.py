import numpy as np
import scipy as sp
import pandas as pd
import scipy.sparse
from sklearn import preprocessing as pp
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import *
from lasagne import layers
from lasagne.updates import nesterov_momentum,adagrad
from sklearn.cross_validation import train_test_split

def make_submission(filename,idx,ypred):
	#generate solution
	preds = pd.DataFrame({"Id": idx, "Hazard": ypred})
	preds = preds.set_index('Id')
	preds.to_csv(filename)
	print ("Storing solution to file %s."%filename)

def expected_scores(y,ypred,n_iter=100,random_state=1):
	expected_public_score = 0
	expected_private_score= 0
	for i in range(n_iter):
		y_private,y_public,ypred_private,ypred_public = train_test_split(y,ypred,test_size=0.3,random_state=random_state+i)
		expected_private_score += gini(y_private,ypred_private)
		expected_public_score += gini(y_public,ypred_public)
	print "expected scores:"
	print "\tpublic  : %f"%(expected_public_score/n_iter)
	print "\tprivate : %f"%(expected_private_score/n_iter)
	
def modify_labels(y,stratergy='padding',padding_function=np.log1p):
	if stratergy=='log':
		return np.log(y*1.)
	elif stratergy=='padding':
		y2 = y.__copy__()
		for i in set(y):
			c = y[y==i].size
			y2[y>i]+=padding_function(c)
		return y2
	else:
		print 'stratergy %s is not supported, return the original labels'%stratergy
	return y

def float32(k):
	return np.cast['float32'](k)

class AdjustVariable(object):
	def __init__(self, name, start=0.02, stop=0.001,decay=0.99):
		self.name = name
		self.start, self.stop = start, stop	
		self.decay = decay
	def __call__(self, nn, train_history):
	
		#epoch = train_history[-1]['epoch']
		current_value = getattr(nn, self.name).get_value()
		if (current_value-self.stop)*(self.start-self.stop)>0:
			new_value =float32(current_value*self.decay)
			getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
	def __init__(self, patience=100,Xvalid=None,yvalid=None,verbose=True,is_regression=True):
		self.patience = patience
		self.best_valid = np.inf
		self.best_valid_epoch = 0		
		self.best_weights = None
		self.Xvalid = Xvalid
		self.yvalid = yvalid
		self.verbose=verbose
		self.is_regression=is_regression
	def __call__(self, nn, train_history):
		ypred_valid = nn.predict(self.Xvalid)
		ypred_valid = 3*ypred_valid+1 if self.is_regression else ypred_valid +1
		current_valid = 1-qkappa(self.yvalid,[int(round(i)) for i in ypred_valid])
		#current_valid = train_history[-1]['valid_loss']
		#train_history[-1]['valid_loss']=current_valid
		valid_mse = ((self.yvalid-ypred_valid)**2).mean()
		current_epoch = train_history[-1]['epoch']
		if self.verbose:			
			print "%04d   |   %5f   |   %5f   |   %5f   |  %4.2f"%(current_epoch,train_history[-1]['train_loss'],valid_mse,1-current_valid,train_history[-1]['dur'])
		if current_valid < self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = nn.get_all_params_values()
		elif self.best_valid_epoch + self.patience < current_epoch:
			print("Early stopping.")
			print("Best valid loss was {:.6f} at epoch {}.".format(
				self.best_valid, self.best_valid_epoch))
			nn.load_params_from(self.best_weights)
			raise StopIteration()

def nn_features(X,y,Xtest,model=None,random_state=1,n_folds=4):
	assert model is not None
	X = X.astype(theano.config.floatX)
	Xtest = Xtest.astype(theano.config.floatX)
	y = y.astype(theano.config.floatX)
	seed = random_state
	nn = build_nn(X.shape[1])
	if nn.output_nonlinearity==sigmoid:
		y = MinMaxScaler().fit_transform(y*1.)
	from lasagne.layers import noise
	from theano.sandbox.rng_mrg import MRG_RandomStreams
	noise._srng =MRG_RandomStreams(seed=random_state)
	try:		
		skf = StratifiedKFold(y, n_folds=n_folds,shuffle=True,random_state=random_state)
		ypred_test = np.zeros(Xtest.shape[0]);
		ypred_train = np.zeros(X.shape[0]);		
		for train_index, test_index in skf:
			seed += 11			
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]			
			y_train = y_train.reshape(-1,1)
			nn = model(input_size=X.shape[1])
			print "%-7s|  %-12s|  %-12s|  %-12s| %-4s"%('epoch','train loss','valid loss','gini','dur')
			nn.on_epoch_finished.append(EarlyStopping(patience=10,Xvalid=X_test,yvalid=y_test,verbose=True))
			np.random.seed(seed)
			nn.fit(X_train,y_train)
			ypred = nn.predict(Xtest).reshape(-1,)
			ypred_valid = nn.predict(X_test).reshape(-1,)
			gini_score = gini(y_test,ypred_valid)
			print "gini score is %f"%(gini_score)
			ypred_test = ypred if ypred_test is None else ypred_test + ypred
			ypred_train[test_index] = ypred_valid
	except KeyboardInterrupt:
		ypred_test = np.zeros(Xtest.shape[0]);
		ypred_train = np.zeros(X.shape[0]);
		return ypred_train, ypred_test
	return ypred_train, ypred_test*1./n_folds

def gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

def evalerror(preds, dtrain):
	labels = dtrain.get_label()
	gini_score = 1- gini(labels,preds)
	return 'gini', gini_score

def count_feature(X, tbl_lst = None, min_cnt = 1):
	X_lst = [pd.Series(X[:, i]) for i in range(X.shape[1])]
	if tbl_lst is None:
		tbl_lst = [x.value_counts() for x in X_lst]
		if min_cnt > 1:
			tbl_lst = [s[s >= min_cnt] for s in tbl_lst]
	X = sp.column_stack([x.map(tbl).values for x, tbl in zip(X_lst, tbl_lst)])
	# NA(unseen values) to 0
	return np.nan_to_num(X), tbl_lst

# mat: A sparse matrix
def remove_duplicate_cols(mat):
	if not isinstance(mat, sp.sparse.coo_matrix):
		mat = mat.tocoo()
	row = mat.row
	col = mat.col
	data = mat.data
	crd = pd.DataFrame({'row':row, 'col':col, 'data':data}, columns = ['col', 'row', 'data'])
	col_rd = crd.groupby('col').apply(lambda x: str(np.array(x)[:,1:]))
	dup = col_rd.duplicated()
	return mat.tocsc()[:, col_rd.index.values[dup.values == False]]

def RImatrix(p, m, k, rm_dup_cols = False, seed = None):
	""" USAGE:
	Argument
	  p: # of original varables
	  m: The length of index vector
	  k: # of 1s == # of -1s
	Rerurn value
	  sparce.coo_matrix, shape:(p, s)
	  If rm_dup_cols == False s == m
	  else s <= m
	"""
	if seed is not None: np.random.seed(seed)
	popu = range(m)
	row = np.repeat(range(p), 2 * k)
	col = np.array([np.random.choice(popu, 2 * k, replace = False) for i in range(p)]).reshape((p * k * 2,))
	data = np.tile(np.repeat([1, -1], k), p)
	mat = sp.sparse.coo_matrix((data, (row, col)), shape = (p, m), dtype = sp.int8)
	if rm_dup_cols:
		mat = remove_duplicate_cols(mat)
	return mat

# Random Indexing
def RI(X, m, k = 1, normalize = True, seed = None, returnR = False):
	R = RImatrix(X.shape[1], m, k, rm_dup_cols = True, seed = seed)
	Mat = X * R
	if normalize:
		Mat = pp.normalize(Mat, norm = 'l2')
	if returnR:
		return Mat, R
	else:
		return Mat

# Return a sparse matrix whose column has k_min to k_max 1s
def col_k_ones_matrix(p, m, k = None, k_min = 1, k_max = 1, seed = None, rm_dup_cols = True):
	if k is not None:
		k_min = k_max = k
	if seed is not None: np.random.seed(seed)
	k_col = np.random.choice(range(k_min, k_max + 1), m)
	col = np.repeat(range(m), k_col)
	popu = np.arange(p)
	l = [np.random.choice(popu, k_col[i], replace = False).tolist() for i in range(m)]
	row = sum(l, [])
	data = np.ones(k_col.sum())
	mat = sp.sparse.coo_matrix((data, (row, col)), shape = (p, m), dtype = np.float32)
	if rm_dup_cols:
		mat = remove_duplicate_cols(mat)
	return mat

def xgb_features(X,y,Xtest,params=None,random_state=0,n_folds=4,early_stop=20,eval_with_gini=False):
	try:
		if params['objective'] == 'reg:logistic':
			yt = MinMaxScaler().fit_transform(y*1.)		
		else:
			yt = y
		skf = StratifiedKFold(yt, n_folds=n_folds,shuffle=True,random_state=random_state)
		ypred_test = np.zeros(Xtest.shape[0])
		ypred_train =np.zeros(X.shape[0])
		seed = random_state;
		dtest = xgb.DMatrix(data=Xtest)
		for train_index, test_index in skf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = yt[train_index], yt[test_index]
			dtrain = xgb.DMatrix(data=X_train,label=y_train)
			dvalid = xgb.DMatrix(data=X_test,label=y_test)
			evallist = [(dtrain,'train'),(dvalid,'valid')]
			num_round = 5000
			params['seed'] = seed+1
			seed+=1
			plst = params.items()
			if eval_with_gini:
				bst = xgb.train( plst, dtrain, num_round,evallist,early_stopping_rounds=early_stop,feval=evalerror)
			else :
				bst = xgb.train( plst, dtrain, num_round,evallist,early_stopping_rounds=early_stop)
			ypred = bst.predict(dtest,ntree_limit=bst.best_iteration)
			ypred_valid = bst.predict(dvalid)
			print ("\tcross validation gini score %s: %f"%(params['objective'],gini(y_test,ypred_valid)))
			ypred_test += ypred
			ypred_train[test_index] = ypred_valid
	except KeyboardInterrupt:
		ypred_test = np.zeros(Xtest.shape[0]);
		ypred_train = np.zeros(X.shape[0]);
		return ypred_train, ypred_test		
	return ypred_train, ypred_test*1./n_folds

def sklearn_features(X,y,Xtest,model,random_state=0,n_folds=4):
	try:
		print (model)
		skf = StratifiedKFold(y, n_folds=n_folds,shuffle=True,random_state=random_state)
		ypred_test = np.zeros(Xtest.shape[0]);
		ypred_train = np.zeros(X.shape[0]);
		seed = random_state;
		for train_index, test_index in skf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			model.random_state=seed
			seed+=1
			model.fit(X_train,y_train)
			ypred = model.predict(Xtest)
			ypred_valid = model.predict(X_test)
			print ("\tcross validation gini score: %f"%gini(y_test,ypred_valid))
			ypred_test = ypred if ypred_test is None else ypred_test + ypred
			ypred_train[test_index] = ypred_valid
	except KeyboardInterrupt:
		ypred_test = np.zeros(Xtest.shape[0]);
		ypred_train = np.zeros(X.shape[0]);
		return ypred_train, ypred_test
	return ypred_train, ypred_test*1./n_folds

def factorizing(train, labels, test,functions=[np.mean,np.var,np.median],keep_old_labels = False):
	train_extended = None
	test_extended = None
	processed_columns = []
	for col in train.columns:
		if type(train[col].values[0]) is str:			
			processed_columns.append(col)
			lbs = set(train[col].values).union(set(test[col].values))
			print "processing column %s with %d labels"%(col,len(lbs))
			for func in functions:
				new_col_train = train[col].values.__copy__()
				new_col_test = test[col].values.__copy__()
				for l in lbs:				
					hazard_values = labels[train[col].values==l]				
					new_col_train[new_col_train==l] = func(hazard_values)
					new_col_test[new_col_test==l] = func(hazard_values)
				train_extended = new_col_train if train_extended is None else np.column_stack((train_extended,new_col_train))
				test_extended = new_col_test if test_extended is None else np.column_stack((test_extended,new_col_test))
			if keep_old_labels:
				lbl = preprocessing.LabelEncoder()
				lbl.fit(lbs)
				train[:,i] = lbl.transform(train[:,i])
				test[:,i] = lbl.transform(test[:,i])
	if not keep_old_labels:
		train = train.drop(processed_columns,axis=1)
		test = test.drop(processed_columns,axis=1)
	print "processed %d columns"%len(processed_columns)
	return np.column_stack((train,train_extended)), np.column_stack((test,test_extended))