import numpy as np 
from sklearn.linear_model import Ridge

def rss(y_true,y_pred):
	"""
	A utility function that returns residual sum of squares
	"""
	return np.sum((y_true-y_pred)**2)

num_splits = 5

# Absolute value below which weights are neglected for reduced features dataset
epsilon = 0.02

"""
Store all errors and coefficients in arrays and find optimum after all iterations
"""
err = []
avg_err = []
reduced_err=[]
coeffs = []
overall_coeffs = []
reduced_coeffs = []

# Vary the regularisation parameter in linear space between 0.01 to 50 
alphas = np.append(np.linspace(0.01,5,num=8,endpoint=False),np.linspace(5,50,num=8))

# Loop for each parameter
for reg_param in alphas:	
	
	for i in range(1,num_splits+1):

		# Read in each split and assign as features and labels
		train = np.genfromtxt("../Dataset/CandC-train"+str(i)+".csv",delimiter=",") 
		test = np.genfromtxt("../Dataset/CandC-test"+str(i)+".csv",delimiter=",")

		x_train,y_train = train[:,:-1],train[:,-1]

		x_test,y_test = test[:,:-1],test[:,-1]  

		# A regularised linear regression model is initialised and fit on the training data

		regressor = Ridge(alpha=reg_param)
		regressor.fit(x_train,y_train)

		# A prediction is made on the test dataset based on the model fitted on train set

		y_pred = regressor.predict(x_test)

		
		coeffs.append(regressor.coef_)
		err.append(rss(y_test,y_pred))
	
	# For each value of regularisation parameter find average error and best fits
	avg_err.append(np.mean(err))
	best_fit_index = np.argmin(err)
	overall_coeffs.append(coeffs[best_fit_index])

# Find regularisation parameter with least average error and print it
best_alpha_index = np.argmin(avg_err)
best_alpha = alphas[best_alpha_index]
best_err = avg_err[best_alpha_index]
best_coeffs = overall_coeffs[best_alpha_index]

print "Best alpha:",best_alpha
print "Error for best alpha:",best_err

# save coefficients as .csv
np.savetxt('coeffs.csv',best_coeffs,delimiter=',')

# Remove features where absolute value is than epsilon
reduced_indices = np.where(np.abs(best_coeffs)>epsilon) 

for i in range(1,num_splits+1):
	# Read in each split and assign features from reduced set and labels

	train = np.genfromtxt("../Dataset/CandC-train"+str(i)+".csv",delimiter=",") 
	test = np.genfromtxt("../Dataset/CandC-test"+str(i)+".csv",delimiter=",")

	x_train,y_train = train[:,reduced_indices[0]],train[:,-1]

	x_test,y_test = test[:,reduced_indices[0]],test[:,-1]  

	# A regularised linear regression model is initialised and fit on the training data

	regressor = Ridge(alpha=best_alpha)
	regressor.fit(x_train,y_train)
	
	# A prediction is made on the test dataset based on the model fitted on train set
	y_pred = regressor.predict(x_test)

	reduced_coeffs.append(regressor.coef_)
	reduced_err.append(rss(y_test,y_pred))
	
# Best amongst the five different splits is reported
print "Average Error with reduced features:",np.mean(reduced_err)
best_reduced_fit_index = np.argmin(reduced_err)

# save coefficients for reduced features as .csv
np.savetxt('coeffs_reduced.csv',reduced_coeffs[best_reduced_fit_index],delimiter=',')

