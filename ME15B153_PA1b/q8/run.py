 # Importing necessary libraries
import numpy as np 
from nn1 import *
from sklearn.preprocessing import StandardScaler

np.random.seed(7)

# Path to data files
dataset_directory = '../DS2/'

# File names for test and train data
train_files = ['Train_features_all','Train_labels_all']
test_files = ['Test_features_all','Test_labels_all']

labels=[0,1,2,3]

# Set of all regularisation parameters to be tested
reg_params = [0.01,0.1,1,10,100]

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

def preprocess(x,y):
	"""
	A utility function that takes as input 'x' which is transformed by 
	Standard Scaler which removes the mean and scales it to unit variance and
	'y' which is one hot encoded 
	"""
	scaler= StandardScaler()
	x=x.astype(np.float64)
	x = scaler.fit_transform(x)
	# One hot encode the target vector
	y = one_hot(y)
	return (x,y)

"""
One hot encode a vector into num_classes
"""
def one_hot(vector,num_classes=4):

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result

"""
Functions to calculate metrics
"""

def accuracy_value(y_true,y_pred):

	err = np.count_nonzero(np.ravel(y_true)-np.ravel(y_pred))
	
	return 1-float(err)/len(y_true)

def precision_value(y_true,y_pred,label=1):

	y_true = np.ravel(y_true)
	y_pred = np.ravel(y_pred)

	pred_pos = np.count_nonzero(y_pred==label)
	tp = np.count_nonzero((y_true==label) & (y_pred==label))

	if pred_pos==0:
		return 0
	else:
		return float(tp)/pred_pos

def recall_value(y_true,y_pred,label=1):

	y_true = np.ravel(y_true)
	y_pred = np.ravel(y_pred)

	cond_pos = np.count_nonzero(y_true==label)
	tp = np.count_nonzero((y_true==label) & (y_pred==label))

	if cond_pos==0:
		return 0
	else:
		return float(tp)/cond_pos


def fscore_value(y_true,y_pred,label=1):

	P = precision_value(y_true,y_pred,label)
	R = recall_value(y_true,y_pred,label)

	if P+R==0:
		return 0
	else:
		return float(2*P*R)/(P+R)

def display_metrics(y_true,y_pred):
	"""
	A function to display various metrics given true
	values,predicted values and label
	"""
	print "Accuracy",accuracy_value(y_true,y_pred)

	for l in labels:
		print "Class",l,"metrics" 
		print "Precision:",precision_value(y_true,y_pred,l)
		print "Recall:",recall_value(y_true,y_pred,l)
		print "F-Measure:",fscore_value(y_true,y_pred,l)

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
x_tr,y_tr = preprocess(x_train,y_train)
x_te,y_te = preprocess(x_test,y_test)

print "### Regularised Error Function ###"
for reg_gamma in reg_params:
	
	nn = NeuralNetwork([96,50,4])
	nn.gradient_descent(x_tr,y_tr,reg_gamma,True)
	print "Gamma",reg_gamma
	y_pred=nn.pred(x_te)
	display_metrics(y_test,y_pred)


print "### Default Error Function"

nn = NeuralNetwork([96,50,4])
nn.gradient_descent(x_tr,y_tr)
y_pred=nn.pred(x_te)
display_metrics(y_test,y_pred)
