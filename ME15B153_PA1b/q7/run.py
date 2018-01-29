"""
Program to run unregularised and regularised 
logistic regression on given dataset
"""

"""
Importing required libraries
subprocess is used to perform command line commands from python
"""
import numpy as np 
import subprocess
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Path to the directory where
dataset_directory = '../DS2/'

# Names of test and train files
train_files = ['Train_features','Train_labels']
test_files = ['Test_features','Test_labels']

# Different labels
labels=[-1,1]

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
	shape = (shape[1],shape[0])

	# Converting values read to integer
	values = np.array(map(int,lines[skip_lines:]))

	if shape[0]!=1:
		return np.reshape(values,shape).T
	else:
		return values

def preprocess(x):
	"""
	A utility function that takes as input 'x' which is transformed by 
	Standard Scaler which removes the mean and scales it to unit variance 
	"""
	scaler= StandardScaler()
	x=x.astype(np.float64)
	x = scaler.fit_transform(x)
	return x

def accuracy_value(y_true,y_pred):
	"""
	A function to calculate accuracy given true and predicted values
	"""
	#No of misclassified points
	err = np.count_nonzero(np.ravel(y_true)-np.ravel(y_pred))
	
	return 1-float(err)/len(y_true)

def precision_value(y_true,y_pred,label=1):
	"""
	A function to calculate precision of a class given true
	values,predicted values and label
	"""
	y_true = np.ravel(y_true)
	y_pred = np.ravel(y_pred)

	# Number of values which were predicted as positive
	pred_pos = np.count_nonzero(y_pred==label)
	
	# Number of true positives
	tp = np.count_nonzero((y_true==label) & (y_pred==label))

	if pred_pos==0:
		return 0
	else:
		return float(tp)/pred_pos

def recall_value(y_true,y_pred,label=1):
	"""
	A function to calculate recall of a class given true
	values,predicted values and label
	"""

	y_true = np.ravel(y_true)
	y_pred = np.ravel(y_pred)

	# Number of values which are positive
	cond_pos = np.count_nonzero(y_true==label)

	# Number of true positives
	tp = np.count_nonzero((y_true==label) & (y_pred==label))

	if cond_pos==0:
		return 0
	else:
		return float(tp)/cond_pos


def fscore_value(y_true,y_pred,label=1):
	"""
	A function to calculate f-measure of a class given true
	values,predicted values and label
	"""
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
x_train,y_train= [read_data(dataset_directory+filename) for filename in train_files]
x_test,y_test= [read_data(dataset_directory+filename) for filename in test_files]

# Preprocess the data
x_train = preprocess(x_train)
x_test = preprocess(x_test)

# Initialise logistic regression model with high C to make it "unregularised"
lr = LogisticRegression(penalty='l2',C=1e9)

# Fit the model to training data
lr.fit(x_train,y_train)

# Predict values on the test data
y_pred = lr.predict(x_test)

"""
Using subprocess.Popen to call the exceutable file l1_logreg_classify
which takes a model,test data and name of the result file as arguments
"""
l1_result_file='result'
subprocess.Popen(['./l1_logreg_classify','model','Test_features',l1_result_file])


# Read data from result file. Here there are 7 lines of headers to be skipped
l1_pred = read_data(l1_result_file,7)


#Display metrics for both cases
print "#### L2 regularised Logistic Regression Results ###"

display_metrics(y_test,y_pred)

print "#### L1 regularised Logistic Regression Results ###"

display_metrics(y_test,l1_pred)