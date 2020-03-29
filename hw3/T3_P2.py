import numpy as np
# might need to install 
import torch
import time

# parameters
N = 2000
M = 100
H = 75

# generate data
np.random.seed(181)
W1 = np.random.normal(scale=0.1, size=(H, M))
b1 = np.random.normal(scale=0.1, size=H)
W2 = np.random.normal(scale=0.1, size=H)
b2 = np.random.normal(scale=0.1, size=1)

X = np.random.random((N, M))
y = np.random.randint(0,2,size=N).astype('float')

# torch copies of data
tW1 = torch.tensor(W1, requires_grad=True)
tb1 = torch.tensor(b1, requires_grad=True)
tW2 = torch.tensor(W2, requires_grad=True)
tb2 = torch.tensor(b2, requires_grad=True)

tX = torch.tensor(X)
ty = torch.tensor(y)

# CAREFUL: if you run the code below w/o running the code above,
# the gradients will accumulate in the grad variables. Rerun the code above
# to reset

# torch implementation
def tforward(X):
  z1 = (torch.mm(tX, tW1.T) + tb1).sigmoid()
  X = (torch.mv(z1, tW2) + tb2).sigmoid()
  return X

tyhat = tforward(tX)
L = -((ty * tyhat.log()) + (1-ty) * (1 - tyhat).log())
# the magic of autograd!
L.sum().backward()

# the gradients will be stored in the following variables
grads_truth = [tW1.grad.numpy(), tb1.grad.numpy(), tW2.grad.numpy(), tb2.grad.numpy()]

# Utils
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# use this to check your implementation
# you can pass in grads_truth as truth and the output of get_grads as our_impl
def compare_grads(truth, our_impl):
  for elt1, elt2 in zip(truth, our_impl):
    if not np.allclose(elt1, elt2, atol=0.001, rtol=0.001):
      return False
  
  return True

# Implement the forward pass of the data. Perhaps you can return some variables that 
# will be useful for calculating the gradients. 
def forward(X):
  z1 = sigmoid(X@W1.T + b1)
  y = sigmoid(z1@W2.T + b2)
  return [z1, y]


# Code the gradients you found in part 2.
# Can pass in additional arguments
def get_grads(y, yhat, X, z): 
  # for loop over data points
  dLdb2 = dLdW2 = dLdb1 = dLdW1 = 0
  for i in range(len(X)):
    dLdb2n = yhat[i] - y[i]
    dLdb2 += dLdb2n

    dLdW2h = z[i]*dLdb2n
    dLdW2 += dLdW2h

    dLdb1n = (1-z[i])*W2*dLdW2h
    dLdb1 += dLdb1n

    dLdb1n = np.resize(dLdb1n, (H, 1))
    Xn = np.resize(X[i], (M, 1))
    dLdW1h = dLdb1n@Xn.T
    dLdW1 += dLdW1h
  # make sure this order is kept for the compare function
  return [dLdW1, dLdb1, dLdW2, dLdb2]

tic = time.perf_counter()
print(compare_grads(grads_truth, get_grads(y, forward(X)[1], X, forward(X)[0])))
toc = time.perf_counter()
print("Vectorized solution took {} seconds".format(toc-tic))