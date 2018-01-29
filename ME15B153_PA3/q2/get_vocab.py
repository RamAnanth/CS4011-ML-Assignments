"""
Utility program that gets all unique words and adds them to the vocabulary
"""
import numpy as np
import os

#  Dataset Directory
dataset_directory = '../Dataset/NaiveBayes/'

vocabs = []

def get_vocab_filewise(filename):
	"""
	Utility function that returns all unique words in file and adds them to 
	vocabulary if they are not there in the vocabulary
	"""
    with open(filename) as file:
		words = []
		for line in file:		
			for word in line.split():
				words.append(word)

    for vocab in np.unique(words)[:-1]:# Till -1 to exclude "Subject:"
		if vocab not in vocabs:
			vocabs.append(vocab)

def get_vocab():


	for dirname in os.listdir(dataset_directory):
	
	#  Searching through all files
		if 'part' not in dirname and '.csv' not in dirname:
			#  Searching through all train and test sets 
			for f in os.listdir(dataset_directory+dirname):
				get_vocab_filewise(dataset_directory+dirname+'/'+f)
	
	return vocabs	
