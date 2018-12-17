import gensim
import smart_open
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from  sklearn.neural_network import MLPClassifier

from  sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

model_path = 'model\\my_doc2vec_model'
model = gensim.models.doc2vec.Doc2Vec.load(model_path)

reviews_test_file = 'data_set\\brown_nltk\\test\\reviews.txt'
religion_test_file = "data_set\\brown_nltk\\test\\religion.txt"
def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)



reviews_list = list(read_corpus(reviews_test_file, True))
religion_list = list(read_corpus(religion_test_file, True))


model_path = 'model\\my_doc2vec_model'
model = gensim.models.doc2vec.Doc2Vec.load(model_path)

X = np.zeros(shape=(1,50))
tags = []
for k in range (0,160):
    vect = model.infer_vector(reviews_list[k])
    X = np.vstack((X,np.array(vect)))
    tags.append(0)

for k in range(0, 160):
    vect = model.infer_vector(religion_list[k])
    X = np.vstack((X, np.array(vect)))
    tags.append(1)

y = np.array(tags)
X = np.delete(X, 0, 0)
# print(X.shape)
# print(y.shape)

kmeans_clusterer =KMeans(n_clusters=2, random_state=0)
y_pred = kmeans_clusterer.fit_predict(X)


db_scan_clusterer = DBSCAN(min_samples=100,  metric='cosine')
y_pred = db_scan_clusterer.fit_predict(X)

plt.plot (y_pred)
plt.show()
print ('kmean score %f' %kmeans_clusterer.score(X) )
print ('dbscan score %f' %db_scan_clusterer.score(X) )

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

y_pred = KMeans(n_clusters=2, random_state=0).fit_predict(X)
plt.plot (y_pred)
plt.show()

# print (reviews_list[0])
# print (model.infer_vector(reviews_list[0]))
# print (religion_list.shape)

# for sent in reviews_list:



