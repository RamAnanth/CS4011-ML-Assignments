# Importing libraries

import numpy as np

np.random.seed(7)# To ensure reproducibility

def sigmoid(x):
    # A function to obtain sigmoid of any input x
        return 1./(1+np.exp(-x))

def sigmoid_prime(x):
    # A function to obtain derivative sigmoid of any input x
    return sigmoid(x)*(1-sigmoid(x))

def softmax(z):
    # A function to compute softmax of any input x
    exp_value = np.exp(z-np.max(z))
    return exp_value / exp_value.sum(axis=1,keepdims=True)

"""
Contains the required class definitions for a 3-layered neural network.
"""

class NeuralNetwork:

    def __init__(self,sizes):

        # Required configuration of [96,50,4] neurons in each layer respectively
        self.sizes = sizes
        l,m,k= self.sizes

        # Random initialization of the model
        self.alpha = np.random.randn(l, m)
        self.b1 = np.random.randn(1, m)
        self.beta = np.random.randn(m,k)
        self.b2 = np.random.randn(1, k)
        
    def pred(self, X):
        """
        Function to predict the label given the input.
        """
        alpha = self.alpha
        b1 = self.b1
        beta = self.beta
        b2 = self.b2

        z1 = X.dot(alpha) + b1
        a1 = sigmoid(z1)

        z2 = a1.dot(beta) + b2

        # Applying softmax
        output = softmax(z2)

         
        # Computing the labels
        labels = np.argmax(output,axis=1)
        return labels

    def cost_derivative(self, output, Y, reg=False):
        """
        Function returns the derivative of cost, given the predicted and the actual values.
        """
        result = np.subtract(output,Y)
        if reg:
            # Compute derivative for alternate error function
            result = np.subtract(output,Y)
            result[Y!=0]*=1-output[Y!=0]
            result[Y==0]*=output[Y==0]

            # result = result*output*(1-output)
        return result

    def gradient_descent(self, X, Y,reg_gamma=0, reg_flag = False, learning_rate = 0.01,num_iter=2000):
        """
        Function to train the neural network using Gradient descent algorithm.
        """
        n_samples = len(Y)
        alpha = self.alpha
        b1 = self.b1
        beta = self.beta
        b2 = self.b2

        # Training for the given no. of iterations ( num_iter )
        for t in range(num_iter):

            z1 = X.dot(alpha) + b1
            a1 = sigmoid(z1)

            z2 = a1.dot(beta) + b2
            output = softmax(z2)

            n_samples = len(X)
 
            grad3 = self.cost_derivative(output,Y,reg_flag)

            # Backpropagation
            
            dbeta = np.matmul(a1.T,grad3)
            db2 = np.sum(grad3,axis=0)

            grad2 = grad3.dot(beta.T)*sigmoid_prime(a1)
            dalpha = np.dot(X.T, grad2)
            db1 = np.sum(grad2,axis=0)

            if reg_flag:
            # Add regularization terms (b1 and b2 don't have regularization terms)
                dbeta += 2 * reg_gamma * beta
                dalpha += 2 * reg_gamma * alpha 

            # Gradient descent parameter update
            self.alpha += -(learning_rate* dalpha)/n_samples
            self.b1 += -(learning_rate * db1)/n_samples
            self.beta += -(learning_rate * dbeta)/n_samples
            self.b2 += -(learning_rate * db2)/n_samples
