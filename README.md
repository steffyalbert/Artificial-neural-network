# Artificial-neural-network
Implementation of an artificial neural network  without using machine learning libraries. 
It has 2 hidden layers and  an input and output layer. Each hidden layer has 5 neurons.  As a loss function we will use the binary cross entropy.

To minimize the loss stochastic gradient descent is used and find the weights that optmize our validation accuracy.

Implemented the backpropagation algorithm to solve this problem. 
The network uses k-fold cross-validation with early stopping to find the best possible hyperparameters. 
• Divide your data set, randomly in two parts:
training set (95%) and validation set (5%).
• Train only on the training set and evaluate the per-example error on the validation set once in a
while, after every 250th iteration.
• Stop training as soon as the error on the validation set is higher than it was the last p times was
checked.
• Use the weights the network had in that previous step as the result of the training run.
