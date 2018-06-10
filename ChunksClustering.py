import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import pickle
np.random.seed(0)

class ChunksClustering:
    def __init__(self):
        # self.affinity_propagation = cluster.AffinityPropagation(damping=.5)
        self.affinity_propagation = pickle.load(open('data/affinity_propagation.pckl', 'rb'))

    def clusterize(self, features):
        features = np.nan_to_num(features)
        self.affinity_propagation.fit(features)
        pickle.dump(self.affinity_propagation, open('data/affinity_propagation.pckl', 'wb'))
        np.save("affinity_propagation.npy", self.affinity_propagation)
        y_pred = self.affinity_propagation.predict(features)
        return y_pred, self.affinity_propagation.cluster_centers_

    def predict_cluster(self, features):
        features = np.nan_to_num(features)
        y_pred = self.affinity_propagation.predict([features])
        dist = distance.euclidean(features,self.affinity_propagation.cluster_centers_[y_pred])
        return y_pred, dist

if __name__ == "__main__":
    c = ChunksClustering()
    l = np.load("music_videos_features.npy")
    print(l.shape)
    print(l[0].shape)
    print(l[0])
    print(l[0].reshape(-1, 1).shape)
    print(c.predict_cluster(l[0]))
