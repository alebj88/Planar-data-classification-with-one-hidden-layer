import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


def layer_sizes(X, Y):
	n_x = X.shape[0]
	n_y = Y.shape[0]
	return n_x, n_y

def initiate_parameters(n_x, n_h, n_y):
	np.random.seed(2)

	W1 = np.random.randn(n_h, n_x) * 0.01
	b1 = np.zeros(shape = (n_h, 1))
	W2 = np.random.randn(n_y, n_h) * 0.01
	b2 = np.zeros(shape = (n_y, 1))

	assert(W1.shape == (n_h, n_x))
	assert(b1.shape == (n_h, 1))
	assert(W2.shape == (n_y, n_h))
	assert(b2.shape == (n_y, 1))

	parameters = {"W1": W1,
					"b1": b1,
					"W2": W2,
					"b2": b2}
	return parameters

def forward_propagation(X, parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	Z1 = np.dot(W1, X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)

	assert(A2.shape == (1, X.shape[1]))
	cache = {"Z1": Z1,
				"A1": A1,
				"Z2": Z2,
				"A2": A2}
	return A2, cache

def backward_propagation(parameters, cache, X, Y):
	m = X.shape[1]

	W1 = parameters["W1"]
	W2 = parameters["W2"]
	A1 = cache["A1"]
	A2 = cache["A2"]

	dZ2 = A2 - Y
	dW2 = (1/m) * np.dot(dZ2, A1.T)
	db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims = True)
	dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1,2))
	dW1 = (1/m) * np.dot(dZ1, X.T)
	db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims = True)

	gradiants = {"dW1": dW1,
					"db1": db1,
					"dW2": dW2,
					"db2": db2}
	return gradiants


def update_parameters(parameters, grads, learning_rate = 1.2):

	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	dW1 = grads["dW1"]
	db1 = grads["db1"]
	dW2 = grads["dW2"]
	db2 = grads["db2"]

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parameters = {"W1": W1,
					"b1": b1,
					"W2": W2,
					"b2": b2}

	return parameters


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
	np.random.seed(3)
	print("Modeling now!")
	n_x, n_y = layer_sizes(X, Y)

	parameters = initiate_parameters(n_x, n_h, n_y)
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]


	for i in range(0, num_iterations):
		A2, cache = forward_propagation(X, parameters)
		gradiants = backward_propagation(parameters, cache, X, Y)
		parameters = update_parameters(parameters, gradiants)

		# if i % 1000 == 0:
		# 	print(i)

	return parameters


def predict(parameters, X):
	print("In predict function!")
	A2, cache = forward_propagation(X, parameters)
	predictions = np.round(A2)
	return predictions

if __name__ == '__main__':
	np.random.seed(1)
	X, Y = load_planar_dataset()
	plt.figure(figsize=(50, 50))
	hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
	for n_h in hidden_layer_sizes:
		parameters = nn_model(X, Y, n_h)
		plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
		predictions = predict(parameters, X)
		accuracy = float((np.dot(Y, predictions.T)) + np.dot(1-Y, 1-predictions.T)) * 100 / float(Y.size)
		print("Accuracy for %d hidden units = " %n_h + str(accuracy))
		plt.show()