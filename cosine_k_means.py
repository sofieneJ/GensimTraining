import numpy as np
import sklearn.metrics as mets
import matplotlib.pyplot as plt
# a = np.array([[10, 7, 4], [3, 2, 1]])
#
# print (np.quantile(a, 0.25, axis=0))
# print (np.quantile(a, 0.75, axis=0))


def load_data ():
    X = np.loadtxt("test_data_X.csv",  delimiter=",")
    y = np.loadtxt("test_data_y.csv", delimiter=",")
    print (X.shape)
    return X, y

def initialize_centroides (X, K):
    feat_dim = X.shape[1]
    centroids = np.zeros(shape=(K, feat_dim))
    if K == 2:
        centroids[0] = np.quantile(X, 0.25, axis=0)
        centroids[1] = np.quantile(X, 0.75, axis=0)

    print (centroids[0].shape)
    return centroids


def assign_data(X, _centroids):
    # dist0 = mets.pairwise_distances(X, _centroids[0].reshape(1,len(_centroids[0])), metric='cosine')
    # dist1 = mets.pairwise_distances(X, _centroids[1].reshape(1,len(_centroids[0])), metric='cosine')
    dist = mets.pairwise_distances(X, _centroids, metric='cosine')
    classes = np.argmin(dist, axis=1)
    #@TODO n_jobs = 4
    # print (classes.shape)
    # unique, unique_indices, unique_counts = np.unique(classes,return_index=True, return_inverse=False, return_counts=True)
    # print (unique)
    # print (unique_indices)
    # print (unique_counts)

    return classes

def compute_centroides (X, K, classes):
    feat_dim = X.shape[1]
    centroids = np.zeros(shape=(K, feat_dim))
    unique = np.unique(classes)
    # index_0 = classes == unique[0]
    for i, classe in enumerate(unique):
        cluster = X[classes == classe]
        print (cluster.shape)
        centroids[i] = np.mean(cluster,axis=0)
    return centroids

def run_consine_cluster (X, K=2):
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

def evaluate_clustering (predicted_classes, actual_classes, method):
    abs_diff_1 = np.absolute(predicted_classes-actual_classes)
    abs_diff_2 = np.absolute(predicted_classes - (1-actual_classes))
    score = 1 - min (np.sum(abs_diff_1)/len(actual_classes), np.sum(abs_diff_2)/len(actual_classes))
    print ('found score for  %s is %f' % (method, score))
    # plt.plot (predicted_classes)
    # plt.show()


def test_consine_clustering ():
    X, y = load_data()
    classes = run_consine_cluster(X)
    evaluate_clustering(classes, y)


# test_consine_clustering()