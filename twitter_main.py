"""
Sample code for auto feature extension

Author: 
Rui Zhao rzhao001@e.ntu.edu.sg
"""

# -*- coding: utf-8 -*-

from io import open
import string
import numpy as np
from gensim import matutils
from numpy import vstack, argsort, float32 as REAL, dot, amax 
import cPickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def save_vocab(vocabs, filepath):
    loc_vocab = open(filepath,'w',encoding='utf-8')
    for word in vocabs:
        loc_vocab.write(word)
        loc_vocab.write(u'\n')
    loc_vocab.close()  

def load_vocab(filepah):
    b_list = []
    with open(filepah,encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            b_list.append(line)
    return b_list


def load_bin_vec(fname):
    """
    Loads 300x1 word vecs from Vector trained from Twitter
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  

    return word_vecs


def vec_vocabular(W_Vecs, vocab):
    """
    Based on local vocab, mapping the WordVectors
    1 Two-grams: additive
    2 Random for OOV
    """   
    k = len(W_Vecs.values()[0])
    W_final = {}
    oov_vocab = set()
    for word in vocab:
        set_word = word.split(' ')
        if len(set_word) == 1:
            if word not in W_Vecs:
                W_final[word] = np.zeros((k,),dtype=float)
                oov_vocab.add(word)
            else:
                W_final[word] = W_Vecs[word]
        else:
            W_final[word] = np.zeros((k,),dtype=float) #for two-gram embeddings
            if set_word[0] in W_Vecs and set_word[1] in W_Vecs:
                W_final[word] = W_Vecs[set_word[0]]+W_Vecs[set_word[1]]
            if set_word[0] in W_Vecs and set_word[1] not in W_Vecs:
                W_final[word] = W_Vecs[set_word[0]]
            if set_word[1] in W_Vecs and set_word[0] not in W_Vecs:
                W_final[word] = W_Vecs[set_word[1]]
    return W_final,oov_vocab




def most_similar(Word2vec, Vocab, positive, topn=100):
        """
        Find the top-N most similar words.

        """
        word_index = Vocab.index(positive) 

        syn0norm = vstack(matutils.unitvec(Word2vec[word]) for word in Vocab).astype(REAL)
        mean = matutils.unitvec(Word2vec[positive]).astype(REAL)
        dists = dot(syn0norm, mean)
        best = argsort(dists)[::-1][:topn + 1]
        # ignore (don't return) words from the input
        result = [(Vocab[sim], dists[sim]) for sim in best if sim != word_index]
        
        return result[:topn]

def most_similar_list(Word2vec, Vocab, positive_list, topn=100):
        """
        Find the top-N most similar words for each word in a list.

        Return K*N: Word Matrix (K is the number of words in the list)

        """
        syn0norm = vstack(matutils.unitvec(Word2vec[word]) for word in Vocab).astype(REAL)
        words_index = set()
        results_all = []
        for positive_word in positive_list:
            words_index.add(Vocab.index(positive_word))
        
        for positive_word in positive_list:     
            mean = matutils.unitvec(Word2vec[positive_word]).astype(REAL)
            dists = dot(syn0norm, mean)
            best = argsort(dists)[::-1][:topn + len(words_index)]
        # ignore (don't return) words from the input
            result = [(Vocab[sim], dists[sim]) for sim in best if sim not in words_index]
            results_all.append(result[:topn])
        return results_all

def weights_base(results_all):
    """
    Based on the cosine similarity, weights each feature

    """
    weights = {}
    for result in results_all:
        for word in result:
            if word[0] in weights:
                weights[word[0]] = amax([weights[word[0]],word[1]]) #updating overlapping feature
            else:
                weights[word[0]]=word[1].astype(REAL)
    return weights

def add_seed(weights,b_list):
    """
    Adding see list, the default weight is 1

    """
    for word in b_list:
        weights[word]= REAL(1)


#loading vocab and bad words
vocab_path = r'data/vocab.txt'
vocab = load_vocab(vocab_path)
blist_path = r'data/bad_v_s.txt'
b_list = load_vocab(blist_path)


W = load_bin_vec("data/word2vec_twitter_model.bin") #WORD EMBEDINGS
W[u'@user'] = W[u'@']
del W[u'@']
W_final, oov = vec_vocabular(W,vocab)


#num_range = [10,50,100,150,200,250,300]
num_range = [50] #
for num in num_range:
    final_feature = most_similar_list(W_final, vocab, b_list, num)

    weights = weights_base(final_feature)

    add_seed(weights,b_list)


#saving_%d' %num_topics) 

    vocab_base = {}
    weights_vec = []
    i = 0
    for key in weights:
        vocab_base[key] = i
        weights_vec.append(weights[key])
        i=i+1

    cPickle.dump([vocab_base, weights_vec], open("vocab_s_%d.p" %num, "wb"))
    print len(vocab_base)
    print "dataset created!"



filepath = 'visual_l_l.txt'

loc_vocab = open(filepath,'w',encoding='utf-8')
for word in vocab_base:
    loc_vocab.write(word)
    loc_vocab.write(u'\t')
    loc_vocab.write(str(weights_vec[vocab_base[word]]).decode('utf-8'))
    loc_vocab.write(u'\n')
loc_vocab.close() 
