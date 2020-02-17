import math
import matplotlib.cm as cm

from math import exp
from CS_181_hw1 import kernel, get_prediction, KERNEL_1, calculate_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']

X = X_df.values
y = y_df.values



def predict_kernel(alpha=0.1):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    kernel_y_preds = []
    for i in range(len(X)):
        kernel_y_preds.append(get_prediction(i, KERNEL_1, alpha, X, y))
    return kernel_y_preds

def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""
    knn_y_preds = []
    for i in range(len(X)):
        distances = {}
        for j in range(len(X)):
            if i != j:
                distances[j] = 1/kernel(X[i], X[j], KERNEL_1)
        
        count = 0
        tot = 0
        for dist in sorted(distances, key=distances.__getitem__):
            if count < k:
                tot += y[dist]
                count += 1
        knn_y_preds.append(tot/count)
    return knn_y_preds


def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')

    # Saving the image to a file, and showing it as well
    plt.savefig(title + '.png')
    plt.show()
    return y_pred

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')

    # Saving the image to a file, and showing it as well
    plt.savefig(title + '.png')
    plt.show()
    return y_pred

for alpha in (0.1, 3, 10):
    print("Loss for alpha = {}: {}").format(alpha, calculate_loss(y, plot_kernel_preds(alpha)))

for k in (1, 5, 15):
    print("Loss for k = {}: {}").format(k, calculate_loss(y, plot_knn_preds(k)))
