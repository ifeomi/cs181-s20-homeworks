import numpy as np
import statistics

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __distance(self, star1, star2):
        return ((star1[0] - star2[0])/3)**2 + (star1[1]-star2[1])**2

    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:
            distances = {}
            for i in range(len(self.X)):
                distances[i] = self.__distance(x, self.X[i])

            vals = []
            count = 0
            
            for dist in sorted(distances, key=distances.__getitem__):
                if count < self.K:
                    vals.append(self.y[dist])
                    count += 1
            try:
                preds.append(statistics.mode(vals))
            except:
                preds.append(vals[0])
        return np.array(preds)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y