# Practice example for DBSCAN clustering.
# This is not optimised (esp. not performance wise), just my first best implementation of DBSCAN
# -1 labels represents noise, positive labels represent clusters

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class dbscan():
    def __init__(self):
        self.labels = None

    def fit(self, X, eps, min_pts):
        
        def set_label(x, cores):

            self.labels[x] = n_cluster
            cores = np.delete(cores, np.argwhere(cores==x))

            close_pts = np.argwhere(distances[x,:] <= eps)
            close_pts = close_pts.reshape(close_pts.shape[0])
            close_pts = close_pts[self.labels[close_pts]==-1]

            for pt in close_pts:
                cores = set_label(pt, cores)
            
            return cores
        
        self.labels = np.ones(X.shape[0]) * (-1)
        distances = distance_matrix(X, X)
        cores = np.argwhere((distances <= eps).sum(axis=0) > min_pts)
        cores = cores.reshape(cores.shape[0])
        n_cluster = 0
        while cores.size != 0:
            cores = set_label(np.random.choice(cores), cores)
            n_cluster +=1

    def labels(self):
        return self.labels

n = 100
data = np.append(np.random.normal(3,1,(round(n/2),2)), np.random.normal(9,1.5,(round(n/2),2)), axis=0)

model = dbscan()
model.fit(data, 1.5, 5)

plt.scatter(data[:,0], data[:,1], c=model.labels)