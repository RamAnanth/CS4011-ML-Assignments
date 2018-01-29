import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

np.random.seed(7) # To ensure repeatability

threshold = 0.5 # Threshold for classification

# Test and train data are read from the Dataset folder
train = np.genfromtxt('../Dataset/DS1-train.csv',delimiter=',')
test = np.genfromtxt('../Dataset/DS1-test.csv',delimiter=',')

np.random.shuffle(train)
np.random.shuffle(test)

"""
All columns except the last column are used as features and 
last column is used as a predictor in both test and training datasets
"""
x_train = train[:,:-1]
y_train = train[:,-1]

x_test = test[:,:-1]
y_test = test[:,-1]

# A linear regression model is initialised and fit on the training data

regressor = LinearRegression()

regressor.fit(x_train,y_train)

# A prediction is made on the test dataset based on the model fitted on train set
y_pred = regressor.predict(x_test)

# All values above the threshold are labelled as 1 and rest as 0
y_pred[y_pred >= threshold] = 1
y_pred[y_pred < threshold] = 0

# Various performance metrics were found and reported
print 'Accuracy:',accuracy_score(y_test,y_pred)
print 'Precision:',precision_score(y_test,y_pred)
print 'Recall:',recall_score(y_test,y_pred)
print 'F1-Score:',f1_score(y_test,y_pred)

# Coefficients learnt by the model are saved in a .csv file. 
np.savetxt('coeffs.csv',regressor.coef_,delimiter=',')
