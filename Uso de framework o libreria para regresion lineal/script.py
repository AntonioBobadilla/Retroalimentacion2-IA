import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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


#obtencion del error de prediccion en test y train
Y_pred = regr.predict(X_test)
Pred_error_test = y_test - Y_pred
Y_pred_train = regr.predict(X_train)
Pred_error_train = y_train - Y_pred_train

print("MSE: ",mean_squared_error(y_test, Y_pred))
print("Model score: ", regr.score(X_test, y_test))

#plot
figure, axis = plt.subplots(2,3)


axis[0,0].scatter(X_test, y_test, alpha = 0.5)
axis[0,0].plot(X_test, Y_pred, color='red')
axis[0,0].set_title("Hours of study vs score (test data)")
axis[0,0].set(xlabel = 'Hours', ylabel = 'Score')
axis[0,0].set_ylim([30,120])


axis[0,1].hist(Pred_error_test, alpha = 0.7, edgecolor = 'black', bins = 20)
axis[0,1].set_title('Histogram of test prediction error')
axis[0,1].set(xlabel = 'Watched prediction error (Y_test - Y_pred)', ylabel = 'Frequency')


axis[0,2].scatter(X_test, y_test, alpha = 0.3, label = 'Real data')
axis[0,2].scatter(X_test, Pred_error_test, color='orange',alpha = 0.1, label = 'Predicted data')
axis[0,2].set_title("Real test data vs Predicted test data")
axis[0,2].set(xlabel = 'Hours', ylabel = 'Score')
axis[0,2].legend()


axis[1,0].scatter(X_train, y_train, alpha = 0.3)
axis[1,0].plot(X_train, Y_pred_train, color='red')
axis[1,0].set_title("Hours of study vs score (train data)")
axis[1,0].set(xlabel = 'Hours', ylabel = 'Score')
axis[1,0].set_ylim([30,120])


axis[1,1].hist(Pred_error_train, alpha = 0.7,edgecolor = 'black', bins = 20)
axis[1,1].set_title('Histogram of train prediction error')
axis[1,1].set(xlabel = 'Watched prediction error (Y_train - Y_pred_train)', ylabel = 'Frequency')


axis[1,2].scatter(X_train, y_train, alpha = 0.3, label = 'Real data')
axis[1,2].scatter(X_train, Pred_error_train, color='orange',alpha = 0.1, label = 'Predicted data')
axis[1,2].set_title("Real train data vs Predicted train data")
axis[1,2].set(xlabel = 'Hours', ylabel = 'Score')
axis[1,2].legend()
plt.show()
