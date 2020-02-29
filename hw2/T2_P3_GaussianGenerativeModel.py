import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.pi = 0
        self.mu = np.array([])
        self.sigma = np.array([])
        self.K = 3

    def __oneHottify(self, y, K):
        return np.eye(K)[y]

    def __piMLE(self, y):
        num_occurrences = y.sum(0)
        return num_occurrences / len(y)

    def __muMLE(self, x, y):
        #Got Guidance from John Tucker
        list = [np.dot(y[:,0], x) / np.sum(y[:, 0]), np.dot(y[:, 1], x) / np.sum(y[:, 1]), np.dot(y[:, 2], x) / np.sum(y[:, 2])]
        return list
    def __sigmaMLE(self, x, y):
        if self.is_shared_covariance == True:
            sum = np.array([])
            for i in range(len(x)):
                #Got guidance from John Tucker
                temp = np.dot(((x[i][np.newaxis]).T - self.mu[0][np.newaxis].T) ,(x[i][np.newaxis].T - self.mu[0][np.newaxis].T).T) + np.dot((x[i][np.newaxis].T - self.mu[1][np.newaxis].T),(x[i][np.newaxis].T - self.mu[1][np.newaxis].T).T) + np.dot((x[i][np.newaxis].T - self.mu[2][np.newaxis].T),(x[i][np.newaxis].T - self.mu[2][np.newaxis].T).T)
                if i == 0:
                    sum = temp
                    print("counter",sum)
                else:
                    sum += temp

            return sum

    def fit(self, X, y):
        y = self.__oneHottify(y, self.K)
        self.pi = self.__piMLE(y)

        self.mu = self.__muMLE(X, y)
        self.sigma = self.__sigmaMLE(X, y)

        print("Sigma", self.sigma)
        print("mu", self.mu)

    def predict(self, X_pred):

        if self.is_shared_covariance==True:
            preds = []
            for x in X_pred:
                temp = np.array([])
                for i in range(self.K):
                    temp = np.append(temp, mvn.pdf(x, self.mu[i], cov=self.sigma))
                preds.append(np.argmax(temp, axis=0))
            return np.array(preds)
        else:
            return

    def negative_log_likelihood(self, X, y):
        pass