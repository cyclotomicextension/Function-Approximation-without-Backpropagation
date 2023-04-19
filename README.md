# Regression using Forward Propagation

# Aim:
The goal of this implementation is to train a regression model to approximate a given function using an alternate approach to backpropagation.

Function Approximation in Regression.ipynb:
The code starts by defining the target function, which we want to approximate. We then generate input values using the numpy library and calculate the corresponding output values using the target function.

Next, we create positive and negative training sets by randomly selecting half of the training data points with positive output values, and half with negative output values. We then combine the positive and negative training sets to create our final training set.

We initialize the weights for the linear regression model using a random normal distribution, and set the learning rate and number of epochs. We then iterate over the training set for the specified number of epochs.

In each iteration, we perform a forward pass to calculate the predicted output values using the current weights. We then calculate the cost using mean squared error between the predicted output values and the actual output values in the training set.

We then perform a forward pass on the positive and negative training sets separately, and compare their mean output values. If the mean output value for the positive training set is greater than the mean output value for the negative training set, we update the weights using the gradient of the mean squared error with respect to the weights and the positive training set. Otherwise, we update the weights using the gradient of the mean squared error with respect to the weights and the negative training set.

Function Approximation in Best Approach to approximate functions.ipynb:
The code starts by defining the target function, which we want to approximate. We then generate input values using the torch library and calculate the corresponding output values using the target function.

Next, we create positive and negative training sets by randomly selecting half of the training data points with positive output values, and half with negative output values. We then combine the positive and negative training sets to create our final training set.

We define a linear regression model with no hidden layers and initialize the weights using a normal distribution. We set the loss function as binary cross-entropy and use stochastic gradient descent (SGD) as the optimizer.

We then iterate over the training set for the specified number of epochs. In each iteration, we perform a forward pass to calculate the predicted output values using the current weights. We then calculate the cost using binary cross-entropy between the predicted output values and the actual output values in the training set.

We then calculate the true positive (tp), false positive (fp), and false negative (fn) rates, and use them to calculate the precision, recall, and F1 score. We use these metrics to evaluate the performance of the model at each epoch.

In each epoch, we also perform an alternate update to the model weights without backpropagation. We do this by using the gradient descent algorithm to update the weights, but instead of calculating the gradients of the loss with respect to the weights, we use a function of the predicted probabilities as the target value for the gradient descent. Specifically, we cube the predicted probabilities and use them as the target value. We also use a threshold of 0.5 to convert the probabilities to binary predictions.

# Author: 
This code is written by me and the documentation is by ChatGPT :p
