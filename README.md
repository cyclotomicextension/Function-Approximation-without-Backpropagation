# Regression using Forward Propagation
 
This is a simple implementation of a regression problem using Forward Propagation. The code uses the numpy library to generate training data, and then trains a simple linear regression model using Forward Propagation.

# Problem Statement

The goal of this problem is to train a model to predict the value of a sine function given a set of input values. 

#Implementation

The code starts by defining the target function as sin(x), which we want to predict. We then generate 500 random input values between -2pi and 2pi using the numpy library, and calculate the corresponding output values using the target function.

Next, we create positive and negative training sets by randomly selecting half of the training data points with positive output values, and half with negative output values. We then combine the positive and negative training sets to create our final training set.

We initialize the weights for the linear regression model using a random normal distribution, and set the learning rate and number of epochs. We then iterate over the training set for the specified number of epochs.

In each iteration, we perform a forward pass to calculate the predicted output values using the current weights. We then calculate the cost using mean squared error between the predicted output values and the actual output values in the training set.

We then perform a forward pass on the positive and negative training sets separately, and compare their mean output values. If the mean output value for the positive training set is greater than the mean output value for the negative training set, we update the weights using the gradient of the mean squared error with respect to the weights and the positive training set. Otherwise, we update the weights using the gradient of the mean squared error with respect to the weights and the negative training set.

# Usage

To run the code, simply execute the python script in a terminal or in an IDE such as Jupyter notebook.

# Author

This code was written by me.
