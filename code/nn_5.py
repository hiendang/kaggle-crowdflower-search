import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from utility import gini, RI,factorizing,xgb_features,modify_labels,sklearn_features
import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum,adagrad
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cross_validation import StratifiedKFold
from lasagne.nonlinearities import *
from utility import gini

def make_submission(filename,idx,ypred):
	#generate solution
	preds = pd.DataFrame({"Id": idx, "Hazard": ypred})
	preds = preds.set_index('Id')
	preds.to_csv(filename)
	print ("Storing solution to file %s."%filename)

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
	return np.asarray(train.todense()),labels.astype(float),np.asarray(test.todense()),idx

def get_data_1(train=None,test=None,functions=[np.mean,np.var]):
	#load train and test 
	if train is None:
		train  = pd.read_csv('../input/train.csv', index_col=0)
	if test is None:
		test  = pd.read_csv('../input/test.csv', index_col=0)
	labels = train.Hazard.values
	train = train.drop('Hazard', axis=1)
	idx = test.index.values
	train,test = factorizing(train,labels,test,functions=functions)
	return train,labels.astype(float),test,idx

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
	def __init__(self, patience=100,Xvalid=None,yvalid=None,verbose=True):
		self.patience = patience
		self.best_valid = np.inf
		self.best_valid_epoch = 0		
		self.best_weights = None
		self.Xvalid = Xvalid
		self.yvalid = yvalid
		self.verbose=verbose		
	def __call__(self, nn, train_history):
		ypred_valid = nn.predict(self.Xvalid)
		current_valid = 1-gini(self.yvalid.reshape(-1,),ypred_valid.reshape(-1,))
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

def build_nn2(input_size=None):
	net1 = NeuralNet(
		layers=[  # three layers: one hidden layer
			('input', layers.InputLayer),
			('dropout0',layers.DropoutLayer),
			('hidden1', layers.DenseLayer),
			('dropout1',layers.DropoutLayer),
			('hidden2', layers.DenseLayer),
			('dropout2',layers.DropoutLayer),
			#('hidden3', layers.DenseLayer),
			#('dropout3',layers.DropoutLayer),
			#('hidden4', layers.DenseLayer),
			#('dropout4',layers.DropoutLayer),
			('output', layers.DenseLayer),
			],
		# layer parameters:
		input_shape=(None, input_size),  # 96x96 input pixels per batch
		dropout0_p=0.2,
		hidden1_num_units=1200,  # number of units in hidden layer
		hidden1_nonlinearity=very_leaky_rectify,
		dropout1_p=0.5,
		hidden2_num_units=800,
		hidden2_nonlinearity=leaky_rectify,
		dropout2_p=0.5,
		#hidden3_num_units=600,
		#hidden3_nonlinearity=leaky_rectify,
		#dropout3_p=0.4,
		#hidden4_num_units=400,
		#hidden4_nonlinearity=sigmoid,
		#dropout4_p=0.3,
		output_nonlinearity=sigmoid,# output layer uses identity function
		output_num_units=1,  # 4 target values
		#objective_loss_function=squared_error,
		#optimization method:
		update=nesterov_momentum,
		#update=adagrad,
		update_learning_rate=theano.shared(float32(0.03)),
		update_momentum=theano.shared(float32(0.9)),
		on_epoch_finished=[
			AdjustVariable('update_learning_rate', start=0.03, stop=0.0005,decay=0.991),
			#EarlyStopping(patience=20),
			AdjustVariable('update_momentum', start=0.8, stop=0.9,decay=1.001),
		],
		regression=True,  # flag to indicate we're dealing with regression problem
		max_epochs=50000,  # we want to train this many epochs
		eval_size=0.0,
		verbose=0,
	)
	return net1

def nn_features(X,y,Xtest,model=build_nn2,random_state=1,n_folds=4,early_stop=20):
	X = X.astype(theano.config.floatX)
	Xtest = Xtest.astype(theano.config.floatX)
	y = y.astype(theano.config.floatX)
	seed = random_state
	nn = model(X.shape[1])
	y = modify_labels(y)
	#y = np.log(y)
	#if nn.output_nonlinearity==sigmoid:
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
			nn.on_epoch_finished.append(EarlyStopping(patience=early_stop,Xvalid=X_test,yvalid=y_test,verbose=True))
			np.random.seed(seed)
			nn.fit(X_train,y_train)
			ypred = nn.predict(Xtest).reshape(-1,)
			ypred_valid = nn.predict(X_test).reshape(-1,)
			gini_score = gini(y_test,ypred_valid)
			print "gini score is %f"%(gini_score)
			if gini_score >0.3:
				ypred_test = ypred if ypred_test is None else ypred_test + ypred
				ypred_train[test_index] = ypred_valid
	except KeyboardInterrupt:
		ypred_test = np.zeros(Xtest.shape[0]);
		ypred_train = np.zeros(X.shape[0]);
		return ypred_train, ypred_test
	return ypred_train, ypred_test*1./n_folds

def exp1(random_state=1):
	train,y,test,idx = get_data_2()
	train = np.log1p(train.astype(float))
	test = np.log1p(test.astype(float))
	scaler = StandardScaler().fit(train)
	train = scaler.transform(train)
	test = scaler.transform(test)
	mtrain = pd.read_csv('meta_features_train.csv')
	mtest = pd.read_csv('meta_features_test.csv')
	scaler2 = StandardScaler().fit(mtrain)
	mtrain = scaler2.transform(mtrain)
	mtest = scaler2.transform(mtest)
	train = np.column_stack((train,mtrain))
	test = np.column_stack((test,mtest))
	rtrain_nn,rtest_nn = nn_features(train,y,test,model=build_nn2,random_state=random_state,n_folds=5,early_stop=50)
	rtrain_nn_total = rtrain_nn
	rtest_nn_total = rtest_nn
	for i in range(9):
		rand_seed = i*7+random_state+1
		rtrain_nn,rtest_nn = nn_features(train,y,test,model=build_nn2,random_state=rand_seed,n_folds=5,early_stop=50)
		pd.DataFrame(data=rtrain_nn_total).to_csv('rtrain_nn_last_4.csv',index=False)
		pd.DataFrame(data=rtest_nn_total).to_csv('rtest_nn_last_4.csv',index=False)
	
	pd.DataFrame(data=rtrain_nn_total/10).to_csv('rtrain_nn_final_4.csv',index=False)
	pd.DataFrame(data=rtest_nn_total/10).to_csv('rtest_nn_final_4.csv',index=False)
	
if __name__ == "__main__":
	exp1(111)
