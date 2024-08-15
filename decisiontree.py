# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:39:55 2021

@author: URAY
"""

import pandas as pd
veriler=pd.read_csv("Breast_cancer_data.csv")

Y=veriler.iloc[:,5:6]
X=veriler.iloc[:,0:5]

#linearregression
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.33, random_state=0)
# from sklearn.linear_model import LinearRegression
# lr=LinearRegression()
# lr.fit(x_train, y_train)
# tahmin=lr.predict(x_test)
# print(tahmin)

# print("evaluation")
# from sklearn.metrics import r2_score
# print(r2_score(y_test, tahmin))

# #SVM-SVR

# from sklearn.svm import SVR
# svr=SVR(kernel="linear")
# svr.fit(x_train,y_train)
# tahmin2=svr.predict(x_test)
# print(tahmin2)

# print("evaluation2")
# from sklearn.metrics import r2_score
print(r2_score(y_test, tahmin2))

from sklearn.svm import SVC
svc=SVC(kernel="rbf")
svc.fit(x_train, y_train)
tahmin_svc=svc.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, tahmin_svc)


# svr2=SVR(kernel="poly", degree=2)
# svr2.fit(x_train,y_train)
# tahmin3=svr2.predict(x_test)
# print(tahmin3)

# print("evaluation3")
# from sklearn.metrics import r2_score
# print(r2_score(y_test, tahmin3))

# svr3=SVR(kernel="rbf")
# svr3.fit(x_train,y_train)
# tahmin4=svr3.predict(x_test)
# print(tahmin4)

# print("evaluation4")
# from sklearn.metrics import r2_score
# print(r2_score(y_test, tahmin4))

# #decision tree-decision tree regression
# from sklearn.tree import DecisionTreeRegressor
# dtr=DecisionTreeRegressor(random_state=0)
# dtr.fit(x_train, y_train)
# tahmin5=dtr.predict(x_test) #neden x test olduğunu sor tekrardan
# print(tahmin5)
# from sklearn import tree
# tree.plot_tree(dtr)

# print("evaluation5")
# from sklearn.metrics import r2_score
# print(r2_score(y_test, tahmin5))


