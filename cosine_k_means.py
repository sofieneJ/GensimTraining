import numpy as np
import sklearn.metrics as mets

# a = np.array([[10, 7, 4], [3, 2, 1]])
#
# print (np.quantile(a, 0.25, axis=0))
# print (np.quantile(a, 0.75, axis=0))




def initialize_centroides (X, K):
    feat_dim = X.shape[1]
    centroids = np.zeros(shape=(K, feat_dim))
    for i in range (0, K):
        centroids[i] = np.quantile(X, (2*i+1)/(2*K), axis=0)

    return centroids


def assign_data(X, _centroids):
    dist = mets.pairwise_distances(X, _centroids, metric='cosine')
    classes = np.argmin(dist, axis=1)
    return classes


def compute_centroides (X, K, classes):
    feat_dim = X.shape[1]
    centroids = np.zeros(shape=(K, feat_dim))
    unique = np.unique(classes)
    # index_0 = classes == unique[0]
    for i, classe in enumerate(unique):
        cluster = X[classes == classe]
        # print (cluster.shape)
        centroids[i] = np.mean(cluster,axis=0)
    return centroids

def run_consine_cluster (X, K):
    print('starting cosine K-means')
    centroids = initialize_centroides(X, K)
    max_iter = 100
    iter=0
    centroids_shift = 1
    while centroids_shift > 0 and iter<max_iter:
        classes = assign_data(X, centroids)
        new_centroids = compute_centroides (X, K, classes)
        centroids_shift = np.linalg.norm(centroids-new_centroids)
        print (' iter % d with centroids distance = %f' %(iter, centroids_shift) )
        iter = iter+1
        centroids = new_centroids

    return classes






# test_consine_clustering(2)
# X, y =  load_data()
# initialize_centroides(X, K=3)