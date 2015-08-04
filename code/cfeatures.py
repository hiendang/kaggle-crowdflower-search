from utility import gini,get_data_3,create_class_features
train,y,test,idx = get_data_3()
ctrain,ctest = create_class_features(train,y,test)
import pandas as pd
dtrain = pd.DataFrame(data=ctrain)
dtrain.to_csv('ctrain.csv',index=False)
dtest = pd.DataFrame(data=ctest)
dtest.to_csv('ctest.csv',index=False)
