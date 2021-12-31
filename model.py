import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data=pd.read_csv(r' C:\Users\Rahavee Prabakaran\Desktop\train.csv')
data.shape
data=pd.get_dummies(data)
data.shape
data.head
train=data[0:7999]
test=data[8000:]
test.head	
data.columns = data.columns.str.strip()
print(data.index.names) 
data.reset_index(drop=True)
x_train=train.drop('Item_Outlet_Sales',axis=1)
y_train=train['Item_Outlet_Sales']
x_test=test.drop(r'Item_Outlet_Sales',axis=1)
true_p=test['Item_Outlet_Sales']
data = data.reset_index()
from sklearn.linear_model import LinearRegression
x_train=pd.get_dummies(x_train)
x_train.shape
x_test=pd.get_dummies(x_test)
x_train.fillna(0,inplace=True)
x_test.fillna(0,inplace=True) 
Ireg=LinearRegression()
Ireg.fit(x_train,y_train)
Ireg.predict(x_test)
Ireg.score(x_test,true_p)
Ireg.score(x_train,y_train)
np.sqrt(np.mean(np.power((np.array(true_p)-np.array(pred)),2)))
np.sqrt(np.mean(np.power((np.array(y_train)-np.array(Ireg.predict(x_train))),2)))
