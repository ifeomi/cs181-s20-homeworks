# CS 181, Spring 2020
# Homework 4

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import distance
from seaborn import heatmap

# This line loads the images for you. Don't change it! 
large_dataset = np.load("data/large_dataset.npy").astype(np.int64)
small_dataset = np.load("data/small_dataset.npy").astype(np.int64)
small_labels = np.load("data/small_dataset_labels.npy").astype(int)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# Keep in mind you may add more public methods for things like the visualization.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.loss = []

    # Private methods
    def __get_dist(self, X1, X2):
        return np.linalg.norm(X1-X2)

    def __get_loss(self, clusters, mus, X):
        loss = 0
        
        for i in range(len(mus)):
            try:
                for point in clusters[i]:
                    loss += self.__get_dist(X[point], mus[i])**2
            except KeyError:
                loss += 0
        return loss

    def __random_init(self, X):
        # randomly assign x's to k
        self.X_indices = [i for i in range(len(X))]
        np.random.shuffle(self.X_indices)  
        size = int(len(self.X_indices)/self.K)
        self.clusters = {}

        for k in range(self.K):
            self.clusters[k] = np.asarray(self.X_indices[size*k : size*(k+1)])
        self.mus = np.zeros((self.K, X.shape[1]))

    def __set_mus(self, X):
        # loop over classes  
        for k in range(self.K):
            # set mu k to average of points in cluster
            try:
                for i in range(len(self.clusters[k])):
                    if i == 0:
                        avg = np.copy(X[self.clusters[k][i]])
                    else:
                        avg = np.sum([avg, X[self.clusters[k][i]]], axis=0)
                self.mus[k] = avg/len(self.clusters[k]) 
            except KeyError:
                pass
        
        self.clusters = {}

    def __set_clusters(self, X):
        for x in self.X_indices:
            # set cluster for each datapoint to the one that min distance
            min_dist = float('inf')
            min_idx = 0
            for i in range(len(self.mus)):
                if self.__get_dist(X[x], self.mus[i]) < min_dist:
                    min_idx = i
                    min_dist = self.__get_dist(X[x], self.mus[i])
            try:
                self.clusters[min_idx] = np.append(self.clusters[min_idx], x)
            except KeyError:
                self.clusters[min_idx] = x

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        self.__random_init(X)

        STEPS = 20
        for i in range(STEPS):
            self.__set_mus(X)
            self.clusters = {}
            self.__set_clusters(X)
            # self.loss.append(self.__get_loss(self.clusters, self.mus, X))
        self.loss = self.__get_loss(self.clusters, self.mus, X)
        ######################################
        ## Problem 2.1:                     ##
        ##                                  ##
        ## K-means objectives function.     ##
        ######################################
        # '''
        plt.plot(range(1, STEPS+1, 1), self.loss, 'bo-')
        plt.xlabel('Number of iterations')
        plt.ylabel('Loss')
        plt.title('Loss vs. number of iterations')
        plt.grid()
        plt.savefig("2_1.png")
        plt.show()
        # '''

######################################
## Problem 2.2:                     ##
##                                  ##
## K-Means Objective w/ diff K's    ##
######################################

# '''
objectives5 = []
objectives10 = []
objectives20 = []
for k in (5, 10, 20):
    KMeansClassifier = KMeans(K=k)
    for i in range(5):
        KMeansClassifier.fit(large_dataset)
        if k == 5:
            objectives5.append(KMeansClassifier.loss)
        elif k == 10:
            objectives10.append(KMeansClassifier.loss)
        else:
            objectives20.append(KMeansClassifier.loss)
        
        print("Loss for k = {}: {}".format(k, KMeansClassifier.loss))

mean_5 = np.mean(objectives5)
mean_10 = np.mean(objectives10)
mean_20 = np.mean(objectives20)

std_5 = np.std(objectives5)
std_10 = np.std(objectives10)
std_20 = np.std(objectives20)

k_vals = ['5', '10', '20']
x_pos = np.arange(len(k_vals))
CTEs = [mean_5, mean_10, mean_20]
error = [std_5, std_10, std_20]

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=5)
ax.set_ylim(1.1*10**10, 1.45*10**10)
ax.set_ylabel('Objective')
ax.set_xticks(x_pos)
ax.set_xticklabels(k_vals)
ax.set_title('K-Means objectives with k = 5, 10, 20')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('2_2.png')
plt.show()
# '''

 ######################################
## Problem 2.3:                       ##
##                                    ##
## K-Means mean images, 5 restarts    ##
 ######################################

# '''
def plot_restarts(data, restarts, title):
    fig = plt.figure()
    fig.suptitle(title)
    
    for i in range(restarts):
        KMeansClassifier.fit(data)
        
        for j in range(len(KMeansClassifier.mus)):
            ax = fig.add_subplot(restarts, K, 1+10*i+j)
            plt.imshow(KMeansClassifier.mus[j].reshape(28,28), cmap='Greys_r')
    plt.savefig("{}.png".format(title))
    plt.show()

RESTARTS = 5

plot_restarts(large_dataset, RESTARTS, '{} random restarts'.format(RESTARTS))
# '''

######################################
## Problem 2.4:                     ##
##                                  ##
## K-Means w/ standardized data     ##
######################################
# '''
standardized_large_dataset = np.copy(large_dataset)
standardized_large_dataset = standardized_large_dataset - np.mean(standardized_large_dataset, axis = 0)
std = [1 if item == 0 else item for item in np.std(standardized_large_dataset, axis = 0)]
standardized_large_dataset = standardized_large_dataset / std

plot_restarts(standardized_large_dataset, RESTARTS, '{} random restarts, standardized data'.format(RESTARTS))
# '''

class HAC(object):
    def __init__(self, linkage):
    	self.linkage = linkage

    def __get_dist(self, X, cluster1, cluster2, linkage):
        if linkage == "min":
            min_dist = float('inf')
            closest_pair = ()
            for idx1 in cluster1:
                for idx2 in cluster2:
                    dist = self.distances[idx1][idx2]
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (idx1, idx2)
            return min_dist
        
        elif linkage == "max":
            max_dist = 0
            furthest_pair = ()
            for idx1 in cluster1:
                for idx2 in cluster2:
                    dist = self.distances[idx1][idx2]
                    if dist > max_dist:
                        max_dist = dist
                        furthest_pair = (idx1, idx2)
            return max_dist
        
        else:
            avg1 = 0
            avg2 = 0
            for i in range(len(cluster1)):
                if i == 0:
                    avg1 = np.copy(X[cluster1[i]])
                else:
                    avg1 = np.sum([avg1,X[cluster1[i]]], axis=0)
            for i in range(len(cluster2)):
                if i == 0:
                    avg2 = np.copy(X[cluster2[i]])
                else:
                    avg2 = np.sum([avg2,X[cluster2[i]]], axis=0) 
            avg1 = avg1 / len(cluster1)
            avg2 = avg2 / len(cluster2)
            return np.linalg.norm(avg1-avg2)

    def fit(self, X):
        # clusters is a nested list. each list within clusters contains the indices within that cluster!
        self.clusters = [[i] for i in range(len(X))]
        
        self.distances = distance.cdist(X, X)
        self.merge_dists = [0]
        # self.merged_clusters = []

        while len(self.clusters) > 10:
            min_dist = float('inf')
            min_idx = ()

            for i in range(len(self.clusters)):
                for j in range(len(self.clusters)):
                    if i != j:
                        # find closest clusters
                        dist = self.__get_dist(X, self.clusters[i], self.clusters[j], self.linkage)  
                        if dist < min_dist:
                            min_idx = (i, j)
                            min_dist = dist
            self.merge_dists.append(min_dist)
            
            self.clusters[min_idx[0]].extend(self.clusters[min_idx[1]])
            del self.clusters[min_idx[1]]
        
        # plt.plot([i for i in range(len(self.merge_dists))], self.merge_dists, '.')
        # plt.xlabel('Total number of merges completed')
        # plt.ylabel('Distance between most recently merged clusters')
        # plt.title('{}-based linkage distance vs. number of merges'.format(self.linkage))
        # plt.grid()
        # plt.savefig("2_6_{}.png".format(self.linkage))
        # plt.show()

    def get_mean_images(self, X):
        means = []
        
        for cluster in self.clusters:
            temp = np.array([])
            for i in range(len(cluster)):
                if i == 0:
                    temp = np.array(X[cluster[i]])

                else:
                    temp = np.sum([temp, X[cluster[i]]], axis=0)
            means.append(temp / len(cluster))
        return means


######################################
## Problem 2.5:                     ##
##                                  ##
## HAC fitting and mean images.     ##
######################################

# '''
title = 'HAC Mean Images'
fig = plt.figure()
fig.suptitle(title)
linkages = ['min', 'max', 'centroid']
for i in range(len(linkages)):
    HACClassifier = HAC(linkages[i])
    HACClassifier.fit(small_dataset)
    for j in range(len(HACClassifier.get_mean_images(small_dataset))):
        ax = fig.add_subplot(3, K, 1 + 10*i + j)
        plt.imshow(HACClassifier.get_mean_images(small_dataset)[j].reshape(28,28), cmap='Greys_r')
plt.savefig("{}.png".format(title))
plt.show()
# '''


######################################
## Problem 2.6:                     ##
##                                  ##
## Distance vs. number of merges.   ##
######################################

# '''
linkages = ['min', 'max', 'centroid']
for linkage in linkages:
    HACClassifier = HAC(linkage)
    HACClassifier.fit(small_dataset)
# '''

######################################
## Problem 2.7:                     ##
##                                  ##
## Confusion matrix heatmap.        ##
######################################

# '''
K = 10
KMeansClassifier = KMeans(K=K)
KMeansClassifier.fit(small_dataset)
kmeans_clusters = KMeansClassifier.clusters

MinHACClassifier = HAC('min')
MinHACClassifier.fit(small_dataset)
min_clusters = MinHACClassifier.clusters

MaxHACClassifier = HAC('max')
MaxHACClassifier.fit(small_dataset)
max_clusters = MaxHACClassifier.clusters

CentroidHACClassifier = HAC('centroid')
CentroidHACClassifier.fit(small_dataset)
centroid_clusters = CentroidHACClassifier.clusters

def plot_heatmap(title, clusters):
    confusion_matrix = np.zeros((K, K))
    real_vals = {}

    for cluster in range(K):
        count = 0
        real_vals[cluster] = []
        for j in range(len(clusters[cluster])):
            real_val = small_labels[clusters[cluster][j]]
            real_vals[cluster].append(real_val)

    for key in real_vals:
        freqs = {}
        for counts in real_vals[key]:
            if counts in freqs:
                freqs[counts] += 1
            else:
                freqs[counts] = 1
        for key2 in freqs:
            confusion_matrix[key][key2] = freqs[key2]
    heatmap(confusion_matrix, annot=True)
    plt.xlabel('True label')
    plt.ylabel('Cluster index')
    plt.title(title)
    plt.savefig("{}.png".format(title))
    plt.show()

plot_heatmap("K-Means Heat Map", kmeans_clusters)
plot_heatmap("Min-Based HAC Heat Map", min_clusters)
plot_heatmap("Max-Based HAC Heat Map", max_clusters)
plot_heatmap("Centroid-Based HAC Heat Map", centroid_clusters)
# '''
            