import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib.patches as mpatches

# Please implement the basis2, basis3, fit, predict methods. 
# Then, create the three plots. An example has been included below.
# You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

# Note: this is in Python 3

def basis1(x):
    return np.stack([np.ones(len(x)), x], axis=1)

def basis2(x):
    return np.stack([np.ones(len(x)), x, x**2], axis=1)

def basis3(x):
    return np.stack([np.ones(len(x)), x, x**2, x**3, x**4, x**5, x**6], axis=1)

class LinearClassifier:
    def __init__(self, eta, steps):
        self.steps = steps
        self.eta = eta
        self.epsilon = 0.000001
        self.loss = 0

    def __sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def __gradient(self, y_hat, x, y):
        return np.dot(x.T, y_hat - y)

    def __getLoss(self, y_hat, y, x):
        for i in range(len(y)):
            # if y[i] == 1:
            #     self.loss += y[i]*np.log(y_hat[i]-self.epsilon)
            # else:
            #     self.loss += (1-y[i])*np.log((1-y_hat[i])-self.epsilon)
            if y_hat[i] < self.epsilon:
                y_hat[i] = self.epsilon
            elif 1-y_hat[i] < self.epsilon:
                y_hat[i] = 1 - self.epsilon
            self.loss += (y[i]*np.log(y_hat[i]) + (1-y[i])*np.log(1-y_hat[i]))
        self.loss = -1*self.loss

    def fit(self, x, y):    
        self.W = np.random.rand(x.shape[1], 1)

        for n in range(self.steps):
            self.W = self.W - self.eta*self.__gradient(self.__sigmoid(np.dot(x, self.W)), x, y)/len(x)
        self.__getLoss(self.__sigmoid(np.dot(x, self.W)), y, x)        

    def predict(self, x, threshold=0.5):
        return self.__sigmoid(np.dot(x, self.W)) >= threshold

# Helps visualize prediction lines
# Takes as input x, y, [list of models], basis function, title
def visualize_prediction_lines(x, y, models, basis, title):
    cmap = c.ListedColormap(['r', 'b'])
    red = mpatches.Patch(color='red', label='Label 0')
    blue = mpatches.Patch(color='blue', label='Label 1')
    plt.scatter(x, y, c=y, cmap=cmap, linewidths=1, edgecolors='black')
    plt.title(title)
    plt.xlabel('X Value')
    plt.ylabel('Y Label')
    plt.legend(handles=[red, blue])

    for model in models:
        X_pred = np.linspace(min(x), max(x), 1000)
        Y_hat = model.predict(basis(X_pred))
        plt.plot(X_pred, Y_hat, linewidth=0.7)

    plt.savefig(title + '.png')
    plt.show()

# Input Data
x = np.array([-8, -3, -2, -1, 0, 1, 2, 3, 4, 5])
y = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)

eta = 0.001
steps = 10000

# TODO: Make plot for each basis with 10 best lines on each plot (accomplished by sorting the models by loss)
# EXAMPLE: Below is plot 10 out of 25 models (weight vector is not optimized yet, completely random)
all_models1 = []
x_transformed1 = basis1(x)

for i in range(25):
    model = LinearClassifier(eta=eta, steps=steps)
    model.fit(x_transformed1, y)
    all_models1.append(model)

all_models1.sort(key=lambda model: model.loss)

visualize_prediction_lines(x, y, all_models1[:10], basis1, "Basis 1 Predictions")

all_models2 = []
x_transformed2 = basis2(x)

for i in range(25):
    model = LinearClassifier(eta=eta, steps=steps)
    model.fit(x_transformed2, y)
    all_models2.append(model)

all_models2.sort(key=lambda model: model.loss)

visualize_prediction_lines(x, y, all_models2[:10], basis2, "Basis 2 Predictions")

all_models3 = []
x_transformed3 = basis3(x)

for i in range(25):
    model = LinearClassifier(eta=eta, steps=steps)
    model.fit(x_transformed3, y)
    all_models3.append(model)

all_models3.sort(key=lambda model: model.loss)

visualize_prediction_lines(x, y, all_models3[:10], basis3, "Basis 3 Predictions")