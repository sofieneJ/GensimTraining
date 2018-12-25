import gensim
from cosine_k_means import *
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

cats = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

test_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball']
# test_cats = ['rec.autos', 'soc.religion.christian']

def read_corpus(corpus, tokens_only=False):
    # with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
    for i, doc in enumerate(corpus):
        if tokens_only:
            yield gensim.utils.simple_preprocess(doc)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), [i])

def load_data():
    data_X_file = '_'.join(test_cats) + '_X.csv'
    data_y_file = '_'.join(test_cats) + '_y.csv'
    X = np.loadtxt(data_X_file,  delimiter=",")
    y = np.loadtxt(data_y_file, delimiter=",")
    # index_shuffle = np.arange(0,X.shape[0])
    # np.random.shuffle(index_shuffle)
    # X = X[index_shuffle]
    # y = y[index_shuffle]
    # print (X.shape)
    return X, y

def select_features():
    model_path = 'model\\my_doc2vec_20news_model'
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)

    X = np.zeros(shape=(1,50))
    tags = []
    for index, category in enumerate(test_cats):
        newsgroup = fetch_20newsgroups(subset='train',
                                             remove=('headers', 'footers', 'quotes'),
                                             categories=[category])
        list_doc = list(read_corpus(newsgroup['data'], True))
        category_size = len(list_doc)
        for k in range(0, category_size):
            vect = model.infer_vector(list_doc[k])
            X = np.vstack((X, np.array(vect)))
            tags.append(index)

    y = np.array(tags)
    X = np.delete(X, 0, 0)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Normalisation
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # clf = LogisticRegression()
    clf = LinearSVC()
    # clf = MLPClassifier()
    clf.fit(X_train, y_train)
    print('LinearSVC clf score %f' % clf.score(X_test, y_test))
    model = SelectFromModel(clf, prefit=True)
    print ('X shape ', X.shape)
    X_new = model.transform(X)
    print ('X_new shape after feature selection', X_new.shape)


    return model


def generate_test_data_set():
    test_dataset_list = []
    for cat in test_cats:
        doc_corpus = list(read_corpus(fetch_20newsgroups(subset='test',
                                          remove=('headers', 'footers', 'quotes'),
                                            categories=[cat])['data'], True))
        test_dataset_list.append(doc_corpus)
         # = len(doc_corpus)

    model_path = 'model\\my_doc2vec_20news_model'
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    X = np.zeros(shape=(1, 50))
    tags = []
    category_sizes = [len(test_dataset_list[k]) for k, _ in enumerate(test_dataset_list)]
    category_size = min (category_sizes)
    for i, _ in enumerate(test_cats):
        for k in range(0, category_size):
            vect = model.infer_vector(test_dataset_list[i][k])
            X = np.vstack((X, np.array(vect)))
            tags.append(i)

    y = np.array(tags)
    X = np.delete(X, 0, 0)

    print (X.shape)
    print (y.shape)

    data_y_file = '_'.join(test_cats) + '_y.csv'
    np.savetxt(data_y_file, y, delimiter=",")

    return X, y



def cluster_with_kmeans(X, K):
    kmeans_clusterer =KMeans(n_clusters=K, random_state=0)
    y_pred = kmeans_clusterer.fit_predict(X)
    # print('kmean score %f' % kmeans_clusterer.score(X))
    # plt.plot (y_pred)
    # plt.show()
    return y_pred

def cluster_with_dbscan(X):
    db_scan_clusterer = DBSCAN(  metric='cosine')
    y_pred = db_scan_clusterer.fit_predict(X)
    return y_pred

def cluster_with_spectral_custering(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    spectral_clusterer = SpectralClustering(n_clusters=2)
    spectral_clusterer.fit(X)
    y_pred = spectral_clusterer.labels_
    return y_pred


def evaluate_clustering(predicted_classes, K, method):
    class_size = int (len(predicted_classes)/K)
    guessed_classes = []
    for i in range (0,K):
        counter = Counter(predicted_classes[i*class_size:(i+1)*class_size])
        guessed_classes.append(counter.most_common(1)[0][0])

    error = 0
    for i in range(0, K):
        predicted_class = np.array(predicted_classes[i*class_size:(i+1)*class_size])
        actual_class = guessed_classes[i]*np.ones(class_size)
        error = error + np.sum(np.absolute(predicted_class-actual_class))

    score = 1 - error/len(predicted_classes)
    print ('found score for  %s is %f' % (method, score))
    fig = plt.figure()
    plt.plot (predicted_classes, '.')
    plt.title('clustering with %s' %method)


data_X_file = '_'.join(test_cats) + '_X.csv'
exists = os.path.isfile(data_X_file)
bForceRegenerate = True
if exists is False or bForceRegenerate is True:
    X, y = generate_test_data_set()
    feature_selector = select_features()
    X_new = feature_selector.transform(X)
    np.savetxt(data_X_file, X_new, delimiter=",")

plt.close('all')
X, y = load_data()
classes = cluster_with_kmeans(X,K=len(test_cats))
evaluate_clustering(classes, K=len(test_cats), method='eulidean K-means')
classes = run_consine_cluster(X,K=len(test_cats) )
evaluate_clustering(classes, K=len(test_cats), method='cosine K-means')
plt.show()

def apply_supervised_classfication(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

    #Normalisation
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print ('LogisticRegression score %f' % clf.score(X_test, y_test))

    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    print ('MLPClassifier score %f' % clf.score(X_test,y_test))

    # REDUCE DIMENSIONS
    pca = PCA(n_components=20)
    X = pca.fit_transform(X)
    print (X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    print ('reduced dimension score %f' % clf.score(X_test,y_test))



# X, y = generate_test_data_set()
# apply_supervised_classfication(X, y)
# cluster_with_kmeans(X)





