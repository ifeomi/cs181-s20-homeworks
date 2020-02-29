import numpy as np



# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.steps = 100000
        self.loss = 0
        self.epsilon = 0.000001


    def __gradient(self, x, y, y_hat):
        return np.dot(x.T, y_hat - y)

    def __oneHottify(self, y, K):
        return np.eye(K)[y]
    def __softmax(self,z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def __updateLoss(self, y, x):
        for i in range(len(y)):
            self.loss -= (np.dot(y[i], np.log(self.__softmax(np.dot(x[i], self.W))) + self.lam/2 * (self.W)**2))
        self.loss = self.loss.sum()

    def fit(self, X, y):
        X = np.stack([np.ones(len(X)), X.T[0], X.T[1]], axis=1)
        y = self.__oneHottify(y, 3)
        self.W = np.random.rand(3, 3)
        # self.__updateLoss(y, X)

        for n in range(self.steps):
            preds = self.__softmax(np.dot(X, self.W))

            self.W -= self.eta*((self.__gradient(X, y, preds)/len(X))+self.lam*self.W)

            # self.__updateLoss(y, X)

            # print(self.loss)
            self.loss = 0


    def predict(self, X_pred):
        X_pred = np.stack([np.ones(len(X_pred)), X_pred.T[0], X_pred.T[1]], axis=1)
        return np.argmax(self.__softmax(np.dot(X_pred, self.W)), axis=1)

    def visualize_loss(self, output_file, show_charts=False):
        pass