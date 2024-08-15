# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:41:58 2021

@author: URAY
"""

import pandas as pd
data=pd.read_csv("healthcare-dataset-stroke-data.csv")
Y=data.iloc[:,-1]

gender=data.iloc[:,1:2].values
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
gender[:,0]=lb.fit_transform(gender[:,0]) #neden 0?

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
gender=ohe.fit_transform(gender).toarray() #neden to array?
gender=pd.DataFrame(data=gender, index=range(5110), columns=["Female","Male","Other"])#index neden girilyor?
#multiple linear - encoder dosyasındaki neden olmuyor onu sor.


married_work_residence=data.iloc[:,5:8]
married_work_residence2=married_work_residence.apply(lb.fit_transform)
#onehot encoder???

smoking=data.iloc[:,10:11].values
smoking[:,0]=lb.fit_transform(smoking[:,0])
smoking=ohe.fit_transform(smoking).toarray()
smoking=pd.DataFrame(data=smoking, index=range(5110), columns=["Unknown","Formerly","Never","Smokes"])

import numpy as np
from sklearn.impute import KNNImputer
nan=np.nan
bmi=data.iloc[:,9:10]
imputer = KNNImputer(n_neighbors=2, weights="uniform")
bmi=imputer.fit_transform(bmi)
bmi=pd.DataFrame(data=bmi, index=range(5110), columns=["bmi"])

df1=data.iloc[:,0:1]
df2=pd.concat([df1, gender], axis=1)
df3=pd.concat([df2, data.iloc[:,2:5]], axis=1)
df4=pd.concat([df3, married_work_residence2], axis=1)
df5=pd.concat([df4, data.iloc[:,8:9]], axis=1)
df6=pd.concat([df5, bmi], axis=1)
X=pd.concat([df6, smoking], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.33, random_state=0)

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(x_train, y_train)
tahmin=dtr.predict(x_test)
print(tahmin)
from sklearn import tree
tree.plot_tree(dtr)


