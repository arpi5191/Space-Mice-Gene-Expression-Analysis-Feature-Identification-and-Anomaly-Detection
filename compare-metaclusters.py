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
        rez1 = cluster_analysis(data, n = 2, method = "kmeans")
        rez2 = cluster_analysis(data, n = 6, method = "kmeans")
        list_of_clusters = list(rez1, rez2)
        m = cluster_meta(list_of_clusters)

def main(inFile = None):
    theMetaClusters = metaClusters(inFile)

    clusterNum1 = int(input("What column do you want to do cluster analysis on?: "))
    clusterNum2 = int(input("What column do you want to do cluster analysis on?: "))

    theMetaClusters.getClusters(clusterNum1, clusterNum2)


if __name__ == "__main__":
    main("breast-cancer-wisconsin.data.csv")
