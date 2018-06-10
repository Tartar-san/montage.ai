import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import pickle
np.random.seed(0)

class ChunksClustering:
    def __init__(self):
        # self.affinity_propagation = cluster.AffinityPropagation(damping=.5)
        self.affinity_propagation = pickle.load(open('data/affinity_propagation.pckl', 'rb'))

    def predictClusters(self, features):
        features = np.nan_to_num(features)
        self.affinity_propagation.fit(features)
        # pickle.dump(self.affinity_propagation, open('data/affinity_propagation.pckl', 'wb'))
        # np.save("affinity_propagation.npy", self.affinity_propagation)
        y_pred = self.affinity_propagation.predict(features)

        return y_pred, self.affinity_propagation.cluster_centers_

if __name__ == "__main__":
    chunks_clustering = ChunksClustering()
    chunks_features = np.load('data/music_videos_features.npy')[0:500]
    print(chunks_features.shape)
    clusters, centers = chunks_clustering.predictClusters(chunks_features)
    print(np.unique(clusters, return_counts=True))
    print(chunks_clustering)
    print(centers)