import math
import numpy as np

# Constants
ALPHA = 10
KERNEL_1 = np.array([[1, 0], [0, 1]])
KERNEL_2 = np.array([[.1, 0], [0, 1]])
KERNEL_3 = np.array([[1, 0], [0, .1]])

# Dataset
targets = np.array([0, 0, 0, .5, .5, .5, 1, 1, 1])
inputs = [[0,0], [0, .5], [0, 1], [.5, 0], [.5, .5], [.5, 1], [1, 0], [1, .5], [1, 1]]
preds_1 = np.array([])
preds_2 = np.array([])
preds_3 = np.array([])

def kernel(x_n, x_star, k, alpha=ALPHA):
	'''Function to find distance between two points'''
	w_matrix = np.dot(alpha, k)
	x = np.dot(np.subtract(x_n, x_star), w_matrix)
	y = np.dot(x, np.transpose([np.subtract(x_n, x_star)]))
	return math.exp(-y)

# Calculate prediction for the input indicated by idx using the other points as training data
def get_prediction(idx, k, alpha, inputs, targets):
	numerator = 0
	for i in range(len(inputs)):
		if i != idx:
			numerator += kernel(inputs[i], inputs[idx], k, alpha)*targets[i]
	
	denominator = 0
	for i in range(len(inputs)):
		if i != idx:
			denominator += kernel(inputs[i], inputs[idx], k, alpha)
	
	return numerator / denominator

def calculate_loss(targets, preds):
	sum_of_squares = ((preds - targets)**2).sum()
	return sum_of_squares

for i in range(len(inputs)):
	preds_1 = np.append(preds_1, get_prediction(i, KERNEL_1, ALPHA, inputs, targets))
	preds_2 = np.append(preds_2, get_prediction(i, KERNEL_2, ALPHA, inputs, targets))
	preds_3 = np.append(preds_3, get_prediction(i, KERNEL_3, ALPHA, inputs, targets))

print("First kernel loss: {}".format(calculate_loss(targets, preds_1)))
print("Second kernel loss: {}".format(calculate_loss(targets, preds_2)))
print("Third kernel loss: {}".format(calculate_loss(targets, preds_3)))
