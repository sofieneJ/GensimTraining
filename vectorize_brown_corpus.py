import gensim
import os
import collections
import smart_open
import random
import time
from nltk.corpus import brown, movie_reviews, reuters

from gensim.test.utils import get_tmpfile



# Set file names for train and test data
train_dir = 'data_set\\brown_nltk\\train\\'
test_dir = 'data_set\\brown_nltk\\test\\'
# test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
brown_train_file = train_dir+ 'train_corpus.txt'
# brown_test_file = test_dir + os.sep + 'lee.cor'

# Define a Function to Read and Preprocess Text
def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


train_corpus = list(read_corpus(brown_train_file))
# test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

print (len(train_corpus[2][0]))

# Training the Model
# Instantiate a Doc2Vec Object
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
# Build a Vocabulary
model.build_vocab(train_corpus)

# Time to Train
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

#save model
# fname = get_tmpfile("model\\my_doc2vec_model")
model_path = 'model\\my_doc2vec_model'
model.save(model_path)
# model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

# Inferring a Vector
# model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
print (model.infer_vector(['I', 'hate', 'lawmakers']) )
