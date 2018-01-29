import numpy as np

np.random.seed(21)# To ensure repeatability

def create_datasets():
	"""
	A utility function that takes the CandC imputed dataset
	and performs 5 different 80-20 splits
	"""

	num_splits = 5

	# Read the imputed dataset
	dataset = np.genfromtxt('../Dataset/CandC_median.csv',delimiter = ',')

	# Number of rows as part of train dataset
	train_test_split_ratio =int(0.8 * dataset.shape[0]) 

	# Loop to split into 5 different datasets
	for i in range(1,num_splits+1):
		
		np.random.shuffle(dataset)
		train = dataset[:train_test_split_ratio,:]
		test = dataset[train_test_split_ratio:,:]
		
		# Save different splits as .csv files in Dataset folder
		np.savetxt("../Dataset/CandC-train"+str(i)+".csv", train, delimiter=",")
		np.savetxt("../Dataset/CandC-test"+str(i)+".csv", test, delimiter=",") 

# Read in the real life dataset
dataset = np.genfromtxt('../Dataset/communities.data',delimiter = ',')

# Remove the non predictive features
dataset = dataset[:,5:] 

# Replacing all nans with -1.0
dataset[np.isnan(dataset)]=-1.0

# Create a copy and use mean imputation
dataset1 = np.copy(dataset)

#Loop that runs for each column in dataset
for i in range(dataset.shape[1]):
	temp = dataset[:,i]
	# Replace missing values with mean of rest of values in the column
	temp[temp==-1.0] = np.mean(temp[temp!=-1.0])

#Loop that runs for each column in dataset
for i in range(dataset1.shape[1]):
	temp = dataset1[:,i]
	
	# Replace missing values with median of rest of values in the column	
	temp[temp==-1.0] = np.median(temp[temp!=-1.0])

# Save imputed datasets in Dataset folder
np.savetxt("../Dataset/CandC_mean.csv", dataset, delimiter=",")
np.savetxt("../Dataset/CandC_median.csv", dataset1, delimiter=",")

# Create 5 different splits
create_datasets()


