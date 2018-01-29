import numpy as np 
import pandas as pd
import os

from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import average_precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

#  Dataset Directory
dataset_directory = '../Dataset/NaiveBayes/'

class NB_Multinomial:
	# Class that implements multinomial Naive Bayes

	def fit(self,X,y):
		# Function that trains a classifier on the given data
		self.cond_prob = np.zeros((2,X.shape[1])) # Conditional probabilites
		self.prior = np.zeros(2)# priors
		
		for c in range(0,2):
			
			X_c = X[y==c]# Documents of class c
			
			self.prior[c] = 1.0*X_c.shape[0]/X.shape[0]

			total_sum = np.sum(X_c) # Total number of tokens
			sums = np.sum(X_c,axis=0)
			sums = sums+1
			V = X_c.shape[1] # Vocabulary size

			self.cond_prob[c] = (1.0*sums/(total_sum+V))

	def predict(self,X):
		# Function that predicts labels for given test data
		scores = np.zeros((X.shape[0],2))
		self.prior = self.prior.reshape(-1,2)
		
		scores = X.dot(np.log(self.cond_prob).T)+np.log(self.prior)

		return np.argmax(scores,axis=1)

	def predict_proba(self,X):
		# Function that return probabilities. These arent exact
		# probability of label

		scores = np.zeros((X.shape[0],2),dtype='float64')
		self.prior = self.prior.reshape(-1,2)
		
		scores = X.dot(np.log(self.cond_prob).T)+np.log(self.prior)

		return np.exp(scores)


class NB_Bernoulli:
# Class that implements Bernoulli Naive Bayes

	def fit(self,X,y):
		# Function that trains classifier on the given data	
		self.cond_prob = np.zeros((2,X.shape[1])) 
		self.prior = np.zeros(2)
		
		X[X>0] = 1 # Changing dataset to indicate only occurence and not number

		for c in range(0,2):
			
			X_c = X[y==c]
			
			N_c = X_c.shape[0]
			
			self.prior[c] = 1.0*N_c/X.shape[0]

			sums = np.sum(X_c,axis=0)
			sums = sums+1

			self.cond_prob[c] = (1.0*sums/(N_c+2))

	def predict(self,X):
		# Function that predicts labels for given test data
		
		X[X>0] =1

		scores = np.zeros((X.shape[0],2))
		self.prior = self.prior.reshape(-1,2)
	 
		scores = X.dot(np.log(self.cond_prob).T)+(1-X).dot(np.log(1-self.cond_prob).T)+np.log(self.prior)

		return np.argmax(scores,axis=1)

	def predict_proba(self,X):
	 	# Function that return probabilities. These arent exact
		# probability of label
		
		X[X>0] =1

		scores = np.zeros((X.shape[0],2),dtype='float64')
		self.prior = self.prior.reshape(-1,2)
	 
		scores = X.dot(np.log(self.cond_prob).T)+(1-X).dot(np.log(1-self.cond_prob).T)+np.log(self.prior)

		return np.exp(scores)

class NB_Beta:

	def __init__(self,alpha,beta):
		self.parameters = [alpha,beta]

	def fit(self,X,y):

		self.cond_prob = np.zeros((2,X.shape[1])) 
		self.prior = np.zeros(2)
		
		X[X>0] = 1

		for c in range(0,2):
			
			X_c = X[y==c]
			
			N_c = X_c.shape[0]
			
			self.prior[c] = 1.0*(N_c+self.parameters[c])/(X.shape[0]+(np.sum(self.parameters)))

			sums = np.sum(X_c,axis=0)
			sums = sums+1

			self.cond_prob[c] = (1.0*sums/(N_c+2))

	def predict(self,X):
		
		X[X>0] =1

		scores = np.zeros((X.shape[0],2))
		self.prior = self.prior.reshape(-1,2)
	 
		scores = X.dot(np.log(self.cond_prob).T)+(1-X).dot(np.log(1-self.cond_prob).T)+np.log(self.prior)

		return np.argmax(scores,axis=1)

	def predict_proba(self,X):
		
		X[X>0] =1

		scores = np.zeros((X.shape[0],2),dtype='float64')
		self.prior = self.prior.reshape(-1,2)
	 
		scores = X.dot(np.log(self.cond_prob).T)+(1-X).dot(np.log(1-self.cond_prob).T)+np.log(self.prior)

		return np.exp(scores)


accuracies = []  
precisions = []  
recalls = []  
fscores = []  
average_precisions = []

for i in range(1,6):

	print "Fold",i

	# Read both test and train data for each fold 
	train_dataset = pd.read_csv(dataset_directory+'train_'+str(i)+'.csv',delimiter = ',',header=None)

	train_dataset = np.array(train_dataset)


	test_dataset = pd.read_csv(dataset_directory+'test_'+str(i)+'.csv',delimiter = ',',header=None)

	test_dataset = np.array(test_dataset)

	clf = NB_Beta(2,10)# Intialise Classifier

	clf.fit(train_dataset[:,:-1],train_dataset[:,-1])# Train the classifier

	y_pred = clf.predict(test_dataset[:,:-1])# Make predictions on test data

	y_pred_proba = clf.predict_proba(test_dataset[:,:-1])
	
	# Get precision and recall values for various thresholds
	precision,recall,_ = precision_recall_curve(test_dataset[:,-1],y_pred_proba[:,1])

	# Plot the PR curve
	plt.plot(recall, precision, color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0, 1.05])
	plt.xlim([0.0, 1.0])
	AP = average_precision_score(test_dataset[:,-1],y_pred_proba[:,1])
	plt.title('Precision-Recall curve: Beta')

	plt.show()

	print 'Accuracy',accuracy_score(test_dataset[:,-1],y_pred)
	accuracies.append(accuracy_score(test_dataset[:,-1],y_pred))
	
	print 'Precision',precision_score(test_dataset[:,-1],y_pred)
	precisions.append(precision_score(test_dataset[:,-1],y_pred))
	
	print 'Recall',recall_score(test_dataset[:,-1],y_pred)
	recalls.append(recall_score(test_dataset[:,-1],y_pred))

	print 'F1-Score',f1_score(test_dataset[:,-1],y_pred)
	fscores.append(f1_score(test_dataset[:,-1],y_pred))

print "Average"
print 'Accuracy',np.mean(accuracies)
	
print 'Precision',np.mean(precisions)
	
print 'Recall',np.mean(recalls)

print 'F1-Score',np.mean(fscores)
