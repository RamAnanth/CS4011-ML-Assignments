import numpy as np
import os

#  Dataset Directory
dataset_directory = '../Dataset/NaiveBayes/'

vocabs = []

def get_word_counts(filename):
"""
Find vocabulary of the training set
"""	
	with open(filename) as file:
		words = []
		for line in file:		
			for word in line.split():
				words.append(word)

	for vocab in np.unique(words)[:-1]:# Till -1 to exclude "Subject:"
		if vocab not in vocabs:
			vocabs.append(vocab)


for dirname in os.listdir(dataset_directory):
#  Searching through all files
	if 'part' not in dirname:
		#  Searching through all train and test sets 
		print dirname
		hams = []
		spams = []
		# print "Total",len(os.listdir(dataset_directory+dirname))
		for f in os.listdir(dataset_directory+dirname):
		# print f	
			get_word_counts(dataset_directory+dirname+'/'+f)
		
			if 'legit' in f:
				hams.append(f)
			# get_word_counts(filename)

			if 'spm' in f:
				spams.append(f)

		print len(hams),len(spams)

# print len(vocabs)

