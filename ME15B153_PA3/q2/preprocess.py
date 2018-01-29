"""
Program that preprocesses the files and saves the feature vectors into test 
and train csv files
"""

import numpy as np
import os

#  Dataset Directory
dataset_directory = '../Dataset/NaiveBayes/'

import get_vocab

vocabulary = get_vocab.get_vocab()

def get_feature(filename):
	"""
	Function that returns feature vector which is a frequency of each word in 
	vocabulary and also adds class labels
	"""
	with open(filename) as file:
		words = []
		for line in file:		
			for word in line.split():
				words.append(word)
	
	unique_words = np.unique(words)[:-1]
	

	feature_vector = np.zeros(len(vocabulary)+1)
	for vocab in unique_words:# Till -1 to exclude "Subject:"
	    if vocab in vocabulary:
	    	feature_vector[vocabulary.index(vocab)] = words.count(vocab)

	if 'spmsg' in filename:
		feature_vector[-1] = 1

	return feature_vector

for dirname in os.listdir(dataset_directory):
#  Searching through all files

	if 'part' not in dirname and '.csv' not in dirname:
		#  Searching through all train and test sets 
		data_list = []
		
		for f in os.listdir(dataset_directory+dirname):
			"""
			Getting feature vector from each file and appending to list
			"""
			feature_vector = get_feature(dataset_directory+dirname+'/'+f)
			data_list.append(feature_vector)

		data_matrix = np.array(data_list)
		np.savetxt(dataset_directory+dirname+'.csv',data_matrix,delimiter=',')

