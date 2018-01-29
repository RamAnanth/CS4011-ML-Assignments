"""
A program to visualise feature extraction by PCA and its decision boundary
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3-D plot of dataset
from matplotlib.axes import Axes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Threshold for classification
threshold = 1.5

# Directory where dataset is stored
dataset_directory = '../Dataset/'
colours=['r','b']
markers = ['o','x']

# Name of test and train files
train_files =['train','train_labels']
test_files =['test','test_labels']

def read_data(filename):
    """
    An utility function that takes as argument the file name and returns dataset as 
    an array
    """
    return np.genfromtxt(dataset_directory+filename+'.csv',delimiter=',')

def lda_transform(x,y):
    """
    A function that transforms a dataset into 1 dimension using LDA given
    features and labels
    """
    lda = LDA(n_components=1)
    
    return lda.fit_transform(x,y)

# Reading dataset
x_train,y_train = [read_data(filename) for filename in train_files]
x_test,y_test = [read_data(filename) for filename in test_files]

# Call function that transforms test,train features into 1D by LDA
x_train1 = lda_transform(x_train,y_train)
x_test1 = lda_transform(x_test,y_test)

# Plotting the data in DS3 as such
fig = plt.figure('Original Data')
ax = fig.add_subplot(111, projection='3d')
ax.set_title('DS3')

for c, m, label in zip(colours,markers,np.unique(y_train)):
    x = x_train[y_train==label]
    xs = x[:,0]
    ys = x[:,1]
    zs = x[:,2]
    ax.scatter(xs, ys, zs,zdir='z',c=c, marker=m)
    ax.legend(['Class 1','Class 2'],loc='upper left')
plt.show()

#Initialising linear regressor
regressor = LinearRegression(fit_intercept=True)

# Fit regressor on train data
regressor.fit(x_train1,y_train)

# Predict values on test data
y_pred = regressor.predict(x_test1)

# Make a prediction based on a certain threshold
y_pred[y_pred>=threshold]=2
y_pred[y_pred<threshold]=1

# Reporting the various performance metrics
print classification_report(y_test,y_pred)
print "Accuracy",accuracy_score(y_test,y_pred) 

# Get the coefficients and intercept of the linear regressor
beta0 = regressor.intercept_
beta1 = regressor.coef_

"""
Plot the extracted data
"""
plt.figure('Transformed Data')
plt.title('LDA transformed DS3')


# Creating uniform variables between max and min values
x_min, x_max = x_train1.min() - 1, x_train1.max() + 1
xx = np.linspace(x_min, x_max, 50)
y_min, y_max = y_pred.min() - 0.2, y_pred.max() + 0.2
yy = np.linspace(y_min, y_max, 50)

# Setting the plot size based on min and max in each direction
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())


for c, m, label in zip(colours,markers,np.unique(y_train)):
    xs = x_train1[y_train==label]
    ys = y_train[y_train==label]	
  
    plt.scatter(xs,ys,c=c,marker=m)

plt.xlabel('Projected dimension values')
plt.ylabel('Y label')

# Find the line fitted 
y_boundary = beta0 + beta1*xx

# Getting the decision boundary based on threshold and line found
decision_x = (threshold - beta0)/beta1

# line1,=plt.plot(xx,y_boundary,'k-',lw=1)
decision_bound,=plt.plot([decision_x]*len(yy),yy,'g-',lw=3)
plt.legend(['Decision boundary','Class 1','Class 2'])
plt.show()
