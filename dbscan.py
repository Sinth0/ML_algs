# Practice example for DBSCAN clustering.
# This is not optimised (esp. not performance wise), just my first best implementation of DBSCAN
# -1 labels represents noise, positive labels represent clusters

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class dbscan():
    def __init__(self):
        self.labels = None

    def fit(self, data, eps, min_pts):

        def label_pt(pt):
            X = data[self.labels==-1]
            self.labels[pt] = n_clusters
            # recursion over all unlabeled points in eps-distance
            #pts_in_reach = [distances[pt,]]
        
        self.labels = np.ones(data.size[0])*(-1)
        distances = distance_matrix(X, X)
        cores = np.argwhere((distances <= eps).sum(axis=0)>min_pts)
        cores = list(cores.reshape(cores.shape[0]))
        n_clusters = 0
        while cores:
            label_pt(pt=np.random.choice(cores))
            # remove all labeled data from cores
            # n_clusters +=1

            


    def labels(self):
        return self.labels

n = 100
data = np.append(np.random.normal(3,1,(round(n/2),2)), np.random.normal(9,1.5,(round(n/2),2)), axis=0)
plt.scatter(data[:,0], data[:,1])

labels = np.ones(data.shape[0])*(-1)
distances = distance_matrix(data,data)
cores = np.argwhere((distances < 1).sum(axis=0)>5)
cores = list(cores.reshape(cores.shape[0]))
(distances[2,:] < 3).sum()