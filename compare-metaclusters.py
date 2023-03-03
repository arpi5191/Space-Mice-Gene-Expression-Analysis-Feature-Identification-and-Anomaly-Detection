import os
import csv
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from sklearn.cluster import KMeans

class metaClusters:

    def __init__(self, inFile):
        self.contents = pd.read_csv(inFile, sep = "\t")

    def getClusters(self, clusterNum1, clusterNum2):
        data = self.contents[clusterNum1:clusterNum2]
        print(data)
        rez1 = cluster_analysis(data, n = 2, method = "kmeans")
        rez2 = cluster_analysis(data, n = 6, method = "kmeans")
        list_of_clusters = list(rez1, rez2)
        m = cluster_meta(list_of_clusters)

# Visualize matrix without reordering
# heatmap(m, Rowv = NA, Colv = NA, scale = "none") # Without reordering
        # numSamples = len(self.contents)
        # range = np.random.RandomState(0)
        # covariance = np.array([[0.4, -0.5], [2.3, 0.9]])
        # cluster_1 = 0.4 * range.randn(numSamples, 2) @ covariance + np.array([2, 2])
        # cluster_2 = 0.3 * range.randn(numSamples, 2) + np.array([-2, -2])
        # print(cluster_meta(cluster_1, cluster_2))
        # data = self.contents[clusterNum1:clusterNum2]
        # cluster_meta(list_of_clusters, rownames = NULL, ...)
        # cluster1 = cluster_analysis(data, n = 3, method = "metacluster")

def main(inFile = None):
    theMetaClusters = metaClusters(inFile)

    clusterNum1 = int(input("What column do you want to do cluster analysis on?: "))
    clusterNum2 = int(input("What column do you want to do cluster analysis on?: "))

    theMetaClusters.getClusters(clusterNum1, clusterNum2)


if __name__ == "__main__":
    main("breast-cancer-wisconsin.data.csv")
