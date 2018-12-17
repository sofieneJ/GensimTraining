from nltk.corpus import brown, movie_reviews, reuters
import numpy as np
from os import listdir
from os.path import isfile, join


# print (brown.categories())
# for cat in brown.categories():
#     print (cat)


# print( reuters.categories())

# print(len(reuters.paras(categories=['housing'])[1]))
def dump_reuters():
    nb_total = 0
    synth_file = open(str('data_set\\'+'synthese_reuters'+'.txt'), 'w')
    for cat in reuters.categories():
    # cat = 'housing'
        text_arr = np.unique(np.array(reuters.paras(categories=[cat])))
        file_object = open(str('data_set\\reuters_nltk\\'+cat+'.txt'), 'w')
        nb_paraph = 0
        for p in range(0, text_arr.shape[0]):
            len_para = 0
            for _, sent in enumerate(text_arr[p]):
                len_para = len_para + len(sent)
            if len_para > 50:
                paragraph = ''
                for i in range(0, len(text_arr[p])):
                    sent = ' '.join(text_arr[p][i])
                    paragraph = str(paragraph) + sent
                file_object.write(paragraph)
                file_object.write('\n')
                nb_paraph = nb_paraph +1
        file_object.close()
        synth_file.write(str('categorie '+cat+' : '+str(nb_paraph)+'\n'))
        nb_total = nb_total + nb_paraph
    synth_file.write(str('Total : ' + str(nb_total) ))
    synth_file.close()


def dump_brown():
    nb_train_total = 0
    nb_test_total = 0
    train_synth_file = open(str('data_set\\brown_nltk\\train\\' + 'synthese'+'.txt'), 'w')
    test_synth_file = open(str('data_set\\brown_nltk\\test\\' + 'synthese' + '.txt'), 'w')
    for cat in brown.categories():
    # cat = 'housing'
        text_arr = np.unique(np.array(brown.sents(categories=[cat])))
        cat_size = int(text_arr.shape[0])
        liste = np.arange(0, cat_size)
        np.random.shuffle(liste)
        test_index = liste[:int(cat_size/5)]
        train_file = open(str('data_set\\brown_nltk\\train\\'+cat+'.txt'), 'w')
        test_file = open(str('data_set\\brown_nltk\\test\\' + cat + '.txt'), 'w')
        nb_train_sent = 0
        nb_test_sent = 0
        for s in range(0, text_arr.shape[0]):
            if s not in test_index and  len(text_arr[s]) > 20:
                sentence = ' '.join(text_arr[s])
                train_file.write(sentence)
                train_file.write('\n')
                nb_train_sent = nb_train_sent +1
            elif len(text_arr[s]) > 20:
                sentence = ' '.join(text_arr[s])
                test_file.write(sentence)
                test_file.write('\n')
                nb_test_sent = nb_test_sent +1
        train_file.close()
        test_file.close()
        train_synth_file.write(str('categorie '+cat+' : '+str(nb_train_sent)+'\n'))
        test_synth_file.write(str('categorie ' + cat + ' : ' + str(nb_test_sent) + '\n'))
        nb_train_total= nb_train_total + nb_train_sent
        nb_test_total = nb_test_total + nb_test_sent
    train_synth_file.write(str('Total : ' + str(nb_train_total) ))
    test_synth_file.write(str('Total : ' + str(nb_test_total)))
    train_synth_file.close()
    test_synth_file.close()

def build_training_corups():
    mypath = 'data_set\\brown_nltk\\train\\'
    corpus_file = open (join(mypath, 'train_corpus.txt'),'a')
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for f in onlyfiles :
        if f is not 'synthese.txt':
            file = open (join(mypath, f), 'r')
            text = file.read()
            corpus_file.write(text)
            file.close()
            # print (join(mypath, f))
    corpus_file.close()


dump_brown()
build_training_corups()

# liste = np.arange(0,20)
# np.random.shuffle(liste)
# print (liste[:5])
# text_arr = np.unique(np.array(brown.sents(categories=['news'])))
# print (text_arr.shape)
# dump_brown()
# corpus = np.array({})
# for cat in reuters.categories():
#     text_arr = np.unique(np.array(reuters.paras(categories=[cat])))
#     corpus = np.unique(np.vstack((corpus, text_arr)))

# print (np.random.randint(low=0, high=50, size=5))
# text = np.unique(np.array(reuters.paras(categories=['housing'])))
# print (len(np.array(text_arr[1]).reshape(-1)))

# print (len(np.array(text_arr[9]).reshape(-1)))
# print (text[3])
# for p in range (0, len(reuters.paras(categories=['housing']))):
#     paragraph = ''
#     for i in range (0, len(reuters.paras(categories=['housing'])[p])):
#         sent = ' '.join(reuters.paras(categories=['housing'])[p][i])
#         paragraph = str(paragraph) + sent
#     file_object.write(paragraph)

