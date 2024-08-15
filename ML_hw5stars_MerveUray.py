# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:41:59 2021

@author: URAY
"""

import pandas as pd
data=pd.read_csv("Stars.csv")
Y=data.iloc[:,-1]


color=data.iloc[:,4:5].values
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
color[:,0]=lb.fit_transform(color[:,0])
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
color=ohe.fit_transform(color).toarray() 
color=pd.DataFrame(data=color, index=range(240), columns=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","r"])

sc=data.iloc[:,5:6].values
sc[:,0]=lb.fit_transform(sc[:,0])
sc=ohe.fit_transform(sc).toarray() 
sc=pd.DataFrame(data=sc, index=range(240), columns=["a","b","c","d","e","f","g"])

df1=data.iloc[:,0:4]
df2=pd.concat([df1, color], axis=1)
X=pd.concat([df2, sc], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.33, random_state=0)

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(x_train, y_train)
tahmin=dtr.predict(x_test)
print(tahmin)
from sklearn import tree
tree.plot_tree(dtr)




