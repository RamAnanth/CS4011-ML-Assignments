import numpy as np 

np.random.seed(7)# To ensure repeatability

"""
The number of features and the number of examples per class
"""
num_features = 20
num_examples = 2000

# The centroid of the first class is generated randomly.
mean1 = np.random.rand(num_features)

"""
Since second centroid must be close to first one 
we create a small vector in random different and add it to first centroid
"""
u = np.random.rand(num_features)

u = u / np.linalg.norm(u) # Normalising the random vector to get a unit vector

epsilon = 2

mean2 = mean1 + epsilon*u

"""
A random matrix is generated and multiplied by it's transponse
to obtain a positive semi-definite and symmetric covariance matrix
"""
rand_matrix = np.random.rand(num_features,num_features)

cov = np.matmul(rand_matrix,rand_matrix.T)

""" 
Two classes are generated from multivariate Gaussian distribution
"""
x1 = np.random.multivariate_normal(mean1, cov, num_examples)
x2 = np.random.multivariate_normal(mean2, cov, num_examples)

# Each class is assigned a label and the label is appended to data
label_1 = np.zeros((num_examples,1),dtype=np.int)
label_2 = np.ones((num_examples,1),dtype=np.int)
x1 = np.hstack((x1,label_1))
x2 = np.hstack((x2,label_2))

"""
Two classes data is shuffled and 70% of each class is put into
the training set and remaining is put into test set
""" 
np.random.shuffle(x1)
np.random.shuffle(x2)

x1_train = x1[:int(x1.shape[0]*0.7)]
x2_train = x2[:int(x2.shape[0]*0.7)]
train = np.vstack((x1_train,x2_train))

x1_test = x1[int(x1.shape[0]*0.7):]
x2_test = x2[int(x2.shape[0]*0.7):]
test = np.vstack((x1_test,x2_test))

# Both train and tests sets are saved to Dataset folder
np.savetxt("../Dataset/DS1-train.csv", train, delimiter=",")
np.savetxt("../Dataset/DS1-test.csv", test, delimiter=",")

print 'Dataset Generated'