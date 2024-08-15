# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 18:20:28 2022

@author: URAY
"""

import pandas as pd
data=pd.read_csv("Breast_cancer_data.csv")

Y=data.iloc[:,5:6]
X=data.iloc[:,0:5]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.33, random_state=0)



from sklearn.linear_model import  LogisticRegression
lr=LogisticRegression()
lr.fit(x_train, y_train)
tahmin_lr=lr.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, tahmin_lr))


from sklearn.svm import SVC
svc=SVC(kernel="rbf")
svc.fit(x_train, y_train)
tahmin_svc=svc.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, tahmin_svc))


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train, y_train)
tahmin_rf=rf.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, tahmin_rf))

