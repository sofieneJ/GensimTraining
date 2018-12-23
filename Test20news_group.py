import gensim
from cosine_k_means import *
import numpy as np
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



def read_corpus(corpus, tokens_only=False):
    # with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
    for i, doc in enumerate(corpus):
        if tokens_only:
            yield gensim.utils.simple_preprocess(doc)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), [i])


def select_features():
    model_path = 'model\\my_doc2vec_20news_model'
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)

    X = np.zeros(shape=(1,50))
    tags = []
    two_catgories = ['rec.autos', 'soc.religion.christian']
    for index, category in enumerate(two_catgories):
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
    print ('X_new shape ', X_new.shape)


    return model

# select_features()

def generate_test_data_set():
    cat_autos = ['rec.autos']
    cat_religion = ['soc.religion.christian']

    newsgroups1 = fetch_20newsgroups(subset='test',
                                      remove=('headers', 'footers', 'quotes'),
                                        categories=cat_religion)

    newsgroups2 = fetch_20newsgroups(subset='test',
                                      remove=('headers', 'footers', 'quotes'),
                                        categories=cat_autos)

    list_doc_1= list(read_corpus(newsgroups1['data'], True))
    list_doc_2 = list(read_corpus(newsgroups2['data'], True))
    model_path = 'model\\my_doc2vec_20news_model'
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    X = np.zeros(shape=(1, 50))
    tags = []
    category_size = min(len(list_doc_1), len(list_doc_2))
    for k in range(0, category_size):
        vect = model.infer_vector(list_doc_1[k])
        X = np.vstack((X, np.array(vect)))
        tags.append(0)

    for k in range(0, category_size):
        vect = model.infer_vector(list_doc_2[k])
        X = np.vstack((X, np.array(vect)))
        tags.append(1)
    y = np.array(tags)
    X = np.delete(X, 0, 0)


    np.savetxt("test_data_y.csv", y, delimiter=",")

    return X, y



def cluster_with_kmeans(X):
    kmeans_clusterer =KMeans(n_clusters=2, random_state=0)
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


exists = os.path.isfile("test_data_X.csv")
if exists is False:
    X, y = generate_test_data_set()
    feature_selector = select_features()
    X_new = feature_selector.transform(X)
    np.savetxt("test_data_X.csv", X_new, delimiter=",")

X, y = load_data()
classes = cluster_with_kmeans(X)
evaluate_clustering(classes, y, method='eulidean K-means')
classes = run_consine_cluster(X)
evaluate_clustering(classes, y, method='cosine K-means')

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





