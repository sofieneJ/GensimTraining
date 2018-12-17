from sklearn.datasets import fetch_20newsgroups
import gensim

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
newsgroups_train = fetch_20newsgroups(subset='train',
                                  remove=('headers', 'footers', 'quotes'),
                                    categories=cats)



print(len(newsgroups_train['data']))


def read_corpus(corpus, tokens_only=False):
    # with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
    for i, doc in enumerate(corpus):
        if tokens_only:
            yield gensim.utils.simple_preprocess(doc)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), [i])


train_corpus = list(read_corpus(newsgroups_train['data']))
#
# print ((train_corpus[1000][0]))
# print (len(train_corpus))

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
# Build a Vocabulary
model.build_vocab(train_corpus)

# Time to Train
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

#save model
# fname = get_tmpfile("model\\my_doc2vec_model")
model_path = 'model\\my_doc2vec_20news_model'
model.save(model_path)
# model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

# Inferring a Vector
# model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
print (model.infer_vector(['I', 'hate', 'lawmakers']) )
