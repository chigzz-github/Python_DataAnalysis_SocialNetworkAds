#simple linear regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values # or y= dataset.iloc[:,3:].values
#splitting the dataset to test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicitng the test results
y_pred = regressor.predict(X_test)
#visualising the training test results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green',)
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#visualising the test test results
plt.scatter(X_test,y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color='brown')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


