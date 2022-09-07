import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# dataset consists of a csv containing hours of study and scores of students. 
data = pd.read_csv('data.csv')


X = np.array(data['study']).reshape(-1, 1) # dependent variable
y = np.array(data['score']).reshape(-1, 1) # independent variable
  
# getting the test and train data from function train_test_split using 50% of the data to the train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)
  
# instantiate an object of LinearRegression from scikit learn
regr = LinearRegression()

# train the model with the train data
regr.fit(X_train, y_train)

# print the accuracy of the test
print(regr.score(X_test, y_test))

#make a prediction based on a test
y_pred = regr.predict(X_test)

# plotting the test and predicted data
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
  
plt.show()
# data scatter of predicted values
