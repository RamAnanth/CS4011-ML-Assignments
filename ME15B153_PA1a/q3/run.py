import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

np.random.seed(7)# To ensure repeatability

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

"""
Store all metrics in arrays and find optimum after all iterations
"""
accuracies = []
precisions = []
recalls = []
fscores = []

# Trying for different values of k
for k in range(3,11):
    
	# A k-NN classification model is initialised and fit on the training data
	clf = KNeighborsClassifier(n_neighbors=k)
	
	clf.fit(x_train,y_train)

	# A prediction is made on the test dataset based on the model fitted on train set
	y_pred = clf.predict(x_test)
	

	# Various performance metrics were found and reported for each k
	print "k =",k
	
	"""
	Report all performance metrics and append values to their respective arrays
	"""
	accuracies.append(accuracy_score(y_test,y_pred))
	print 'Accuracy:',accuracy_score(y_test,y_pred)
	
	precisions.append(precision_score(y_test,y_pred))
	print 'Precision:',precision_score(y_test,y_pred)
	
	recalls.append(recall_score(y_test,y_pred))
	print 'Recall:',recall_score(y_test,y_pred)
	
	fscores.append(f1_score(y_test,y_pred))
	print 'F1-Score:',f1_score(y_test,y_pred)

# Find best fit k based on their accuracies
best_k_index = np.argmax(accuracies)

"""
Display performance metrics for best fit k value
"""
print "Best fit k =",best_k_index + 3
print 'Best fit Accuracy:',accuracies[best_k_index]	
print 'Best fit Precision:',precisions[best_k_index]
print 'Best fit Recall:',recalls[best_k_index]
print 'Best fit F1-Score:',fscores[best_k_index]