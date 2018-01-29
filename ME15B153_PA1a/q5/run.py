import numpy as np 
from sklearn.linear_model import LinearRegression

def rss(y_true,y_pred):
	"""
	A utility function that returns residual sum of squares
	"""
	return np.sum((y_true-y_pred)**2)

num_splits = 5

"""
Store all errors and coefficients in arrays and find optimum after all iterations
"""
err = []
coeffs = []

# Loop for different splits
for i in range(1,num_splits+1):
	
	# Read in each split and assign as features and labels
	train = np.genfromtxt("../Dataset/CandC-train"+str(i)+".csv",delimiter=",") 
	test = np.genfromtxt("../Dataset/CandC-test"+str(i)+".csv",delimiter=",")

	x_train,y_train = train[:,:-1],train[:,-1]  
	x_test,y_test = test[:,:-1],test[:,-1]  
	
	# A linear regression model is initialised and fit on the training data
	regressor = LinearRegression()
	regressor.fit(x_train,y_train)
	
	# A prediction is made on the test dataset based on the model fitted on train set
	y_pred = regressor.predict(x_test)
	
	# Append coefficients and error values
	coeffs.append(regressor.coef_)
	err.append(rss(y_test,y_pred))
	
print 'Average Residual Error:',np.mean(err)

# Choose the split with lowest residual sum of squares
best_fit_index = np.argmin(err)

# save coefficients as .csv
np.savetxt('coeffs.csv',coeffs[best_fit_index],delimiter=',')



