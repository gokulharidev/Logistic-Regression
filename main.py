from LogisticRegression import Logistic

import pandas as pd
file=pd.read_csv("samdataset.csv")
x=file.iloc[:,0]
y=file.iloc[:,1]

model=Logistic()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
loss=model.fit(x_train,y_train)
print("Final Loss:",loss)
y_pred=model.predict(x_test)
print("Accuracy:",model.accuracy(y_test,y_pred))
