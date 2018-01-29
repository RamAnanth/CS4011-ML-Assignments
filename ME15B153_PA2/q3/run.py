"""
A program to visualise the boundaries learnt by the different discriminant analysis
algorithms like LDA and QDA on the iris dataset
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import LabelEncoder # To Encode labels as numbers

# Directory where dataset is stored
dataset_directory = '../Dataset/'
colours=['r','b','g']
markers = ['o','x','^']
labels = ['Iris-setosa','Iris-versicolor','Iris-virginica']

def read_data(filename):
    """
    An utility function that reads dataset as string to ensure that class labels
    are read.
    """
    return np.genfromtxt(dataset_directory+filename,delimiter=',',dtype=str)

dataset = read_data('iris.csv')

le = LabelEncoder()

# Using only petal length and width as features
X = dataset[:,2:4]
Y = dataset[:,-1]

# Encode class labels as numbers
Y = le.fit_transform(Y)

# Convert features to float as they are read as string 
X = X.astype(np.float)

# Initialising the classifiers
lda = LDA()
qda = QDA()

# Fit the data to the classifiers.
lda.fit(X, Y)
qda.fit(X, Y)

# Creating variables that takes value that cover range of the training data
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.03), np.arange(y_min, y_max, 0.01))

# Predict a value for the points obtained above
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Zq = qda.predict(np.c_[xx.ravel(), yy.ravel()])

Zq = Zq.reshape(xx.shape)
Z = Z.reshape(xx.shape)

plt.figure()

# Plot the values and their predictions as colored mesh to get visible boundaries
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.title('LDA')
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Petal length')
plt.ylabel('Petal width')

# Setting the plot size based on min and max in each direction
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()

plt.figure()
plt.title('QDA')
# Plot the values and their predictions as colored mesh to get visible boundaries
plt.pcolormesh(xx, yy, Zq, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Petal length')
plt.ylabel('Petal width')

# Setting the plot size based on min and max in each direction
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
