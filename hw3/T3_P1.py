import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math

D = [3.3, 3.5, 3.1, 1.8, 3.0, 0.74, 2.5, 2.4, 1.6, 2.1, 2.4, 1.3, 1.7, 0.19]
SIGMA_SQUARED = 1
SIGMA = math.sqrt(SIGMA_SQUARED)
TAU_SQUARED = 5
TAU = math.sqrt(TAU_SQUARED)
n = len(D)

def mle_predictor(x, data):
	mu_mle = sum(data)/len(data)
	return norm.pdf(x, mu_mle, SIGMA_SQUARED)

def map_predictor(x, data):
	mu_map = sum(data)/(len(data)+SIGMA_SQUARED/TAU_SQUARED)
	return norm.pdf(x, mu_map, SIGMA_SQUARED)

def posterior_predictive(x, data):
	sig_squared_n = 1/(1/TAU_SQUARED+len(data)/(SIGMA_SQUARED))
	mu_n = sig_squared_n*(sum(data)/SIGMA_SQUARED)
	return norm.pdf(x, mu_n, sig_squared_n+SIGMA_SQUARED)

def graph():
	# After you implement the first three functions, this will graph your pdfs
	fig, ax = plt.subplots(nrows=5, ncols=3)
	x = np.arange(-8, 8, 0.1)
	data_idx = 1
	for row in ax:
		for col in row:
			trimmed_data = D[:data_idx]
			col.plot(x, mle_predictor(x, trimmed_data), alpha=0.5)
			col.plot(x, map_predictor(x, trimmed_data), alpha=0.5)
			col.plot(x, posterior_predictive(x, trimmed_data), alpha=0.5)
			col.set_title(f"{data_idx} points")
			data_idx += 1
	fig.legend(["MLE", "MAP", "Posterior Predictive"])
	fig.subplots_adjust(hspace=0.75) # Adjust this if your title are overlapping with graphs
	plt.savefig('1_3.png')
	plt.show()

def marginal_likelihood():
	c = SIGMA/(((SIGMA*math.sqrt(2*math.pi))**n)*math.sqrt(n*TAU_SQUARED+SIGMA_SQUARED))
	D_squared = [i**2 for i in D]
	exp1 = math.exp(-sum(D_squared)/(2*SIGMA_SQUARED))
	exp2 = math.exp(sum(D)**2/(2*SIGMA_SQUARED*(n+SIGMA_SQUARED/TAU_SQUARED)))
	print(c*exp1*exp2)
	# If you want, you can use this function to calculate the answers for 5 and 6

if __name__ == "__main__":
	graph()
	marginal_likelihood()
