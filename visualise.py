import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualise(data, outliers_idx):
    if data.shape[1] > 50:
        pca = PCA(n_components = 50)
        pca.fit(data[outliers_idx==1])
        data = pca.transform(data)
    data = TSNE(n_components=2).fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    outliers = data[outliers_idx == -1]
    inliers = data[outliers_idx == 1]
    ax.scatter(inliers[:, 0], inliers[:, 1], c='b', label='inliers')
    ax.scatter(outliers[:, 0], outliers[:, 1], c='r', label='outliers')
    plt.show()
        
