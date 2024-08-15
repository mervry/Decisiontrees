# -*- coding: utf-8 -*-
"""


@author: URAY
"""

import pandas as pd
data=pd.read_csv("bodyfat.csv")

Y=data.iloc[:,1:2]
df1=data.iloc[:,0:1]
X=pd.concat([df1, data.iloc[:,2:16]], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.33, random_state=0)

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(x_train, y_train)
tahmin=dtr.predict(x_test)
print(tahmin)
from sklearn import tree
tree.plot_tree(dtr)
