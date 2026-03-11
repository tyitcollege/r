import numpy as np
from sklearn.cluster import KMeans

x = np.array([[15,39],[16,81],[17,6],[18,77],[19,40],[20,80]])

kmeans = KMeans(n_clusters=3).fit(x)
print("Cluster labels:", kmeans.labels_)
