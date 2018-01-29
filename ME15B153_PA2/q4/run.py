"""
A Program to tune some hyperparameters of the different kernel types in SVM 
"""
from sklearn.svm import SVC
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib # To save the models

np.random.seed(7)

# Path to data files
dataset_directory = '../Dataset/'

# File names for test and train data
train_files = ['Train_features_all','Train_labels_all']
test_files = ['Test_features_all','Test_labels_all']

labels=[0,1,2,3]

def read_data(filename,skip_lines=2):
	
	"""
	A utility function that opens a file reads it and 
	returns a required data array.
	skip_lines is number of lines in header after which we have required data
	"""

	with open(filename) as f:
		lines = [line.strip('\n') for line in f.readlines()]
	
	# Getting shape of the data from the (skip_lines -1)th line 

	shape=tuple(map(int,lines[skip_lines-1].split()))
	
	# Converting values read to integer
	values = np.array(map(int,lines[skip_lines:]))
	shape = (shape[1],shape[0])

	if shape[0]!=1:
		return np.reshape(values,shape).T
	else:
		return values.reshape(-1,1)

def preprocess(x):
	"""
	A utility function that takes as input 'x' which is transformed by 
	Standard Scaler which removes the mean and scales it to unit variance.
	"""
	scaler= StandardScaler()
	x=x.astype(np.float64)
	x = scaler.fit_transform(x)
	return x

# Read in test and train data 
x_train,y_train = [read_data(dataset_directory+filename) for filename in train_files]
x_test,y_test= [read_data(dataset_directory+filename) for filename in test_files]

# Shuffle test and train data
train_data=np.hstack((x_train,y_train))
np.random.shuffle(train_data)

x_train,y_train=train_data[:,:-1],train_data[:,-1]

test_data=np.hstack((x_test,y_test))
np.random.shuffle(test_data)

x_test,y_test=test_data[:,:-1],test_data[:,-1]

# Preprocess data
x_train = preprocess(x_train)
x_test = preprocess(x_test)

# Setting range of values to be tried out for different parameters
C_range = np.logspace(-7,4,num=9,base=2)
gamma_range = np.logspace(-7,4,num=8,base=2)
kernels = ['linear','poly','rbf','sigmoid']
degrees_range = [i for i in range(2,6)]
coef0_range = [i for i in range(0,5)]

# Setting grids for each kernel based on parameters needed
param_grid_poly = dict(gamma=gamma_range,C=C_range,degree=degrees_range)
param_grid_sigmoid = dict(gamma=gamma_range,C=C_range,coef0=coef0_range)
param_grid_rbf = dict(gamma=gamma_range,C=C_range)

print "Linear Kernel-------------------------------"

# Store all validation accuracies
accuracies = []

# Loop for all values to tune
for C in C_range:
	svc = SVC(kernel='linear',C=C)
	
	# Obtain 5 fold cross validation score
	scores = cross_val_score(svc, x_train,y_train, cv=5)
	svc.fit(x_train,y_train)
	print "C:",C
	print "Accuracy:",scores.mean()
	accuracies.append(scores.mean())

# Plotting Accuracy vs logC 
plt.figure()
plt.title('Linear Kernel Parameter Evaluation')
plt.xlabel('log C')
plt.ylabel('Mean Accuracies from Cross Validation')
plt.plot(np.log2(C_range),accuracies,'ro-')
plt.show()

# Finding best parameters and values
best_idx = np.argmax(accuracies)
svc = SVC(kernel = 'linear',C=C_range[best_idx])
svc.fit(x_train,y_train)

joblib.dump(svc,'svm_model1.model')

print "--------------------------------------------"

print "polynomial Kernel-------------------------------"

svc = SVC(kernel='poly',coef0=1)

# Run a grid search over all parameters for polynomial Kernel
clf = GridSearchCV(svc, param_grid_poly,cv=5)
clf.fit(x_train,y_train)
print clf.best_params_
print clf.best_score_
print "--------------------------------------------"
joblib.dump(clf,'svm_model2.model')


print "Radial Basis Function Kernel-------------------------------"

svc = SVC(kernel='rbf')

# Run a grid search over all parameters for rbf Kernel
clf = GridSearchCV(svc, param_grid_rbf,cv=5)
clf.fit(x_train,y_train)
print clf.best_params_
print clf.best_score_
print "--------------------------------------------"

joblib.dump(clf,'svm_model3.model')

print "Sigmoid Kernel-------------------------------"

svc = SVC(kernel='sigmoid')

# Run a grid search over all parameters for sigmoid Kernel
clf = GridSearchCV(svc, param_grid_sigmoid,cv=5)
clf.fit(x_train,y_train)
print clf.best_params_
print clf.best_score_
print "--------------------------------------------"
joblib.dump(clf,'svm_model4.model')
