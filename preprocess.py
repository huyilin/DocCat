# -*- coding: utf-8 -*-
"""
Sample code for twitter processing

Author: 
Rui Zhao rzhao001@e.ntu.edu.sg
"""



from io import open
from bs4 import BeautifulSoup     
import glob
import string
import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from twokenize import tokenizeRawTweetText 
from gensim import matutils
from numpy import vstack, argsort, float32 as REAL, dot
import cPickle


del_list = [u' ‘', u'›', u'“', u'x']      #predefined numbers  


def replace_at(contentlist):
    indices_to_replace = [i for i,x in enumerate(contentlist) if u'@' in x]
    for i in indices_to_replace:
        contentlist[i] = u'@USER'
    return contentlist


def message_to_words( message, word_del):
    message_clean = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "URL", message)    
    word_tokens=tokenizeRawTweetText (message_clean)              # tokenize
    word_tokens = replace_at(word_tokens)  #replace @ using predefined char
    word_tokens=[s for s in word_tokens if s not in word_del]   # remove few speicific words "[" "]" 
    #word_list1 = [w.lower() for w in word_tokens if not w.lower() in nltk.corpus.stopwords.words('english') and not w in string.punctuation] #lowering the case and stop words removal
    word_list1 = [w.lower() for w in word_tokens if not w in string.punctuation]
    word_list2 = [x for x in word_list1 if not (x[0].isdigit() or x[0] == '-' and x[1:].isdigit())] # remove digit no.            
    #word_stemmed=[porter.stem(t) for t in word_list2]   # stemming to get normalized words       
    #word_stemmed1= ' '.join(str(v).encode('ascii', 'ignore') for v in word_stemmed)   # s.endcode('ascii','ignore') avoid the utf8 decode errors     
    return word_list2

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



#defining global variables
corpus_path = 'datasets//tweetID_text.txt'
messages = []
index_list = []
processed_corpus=[]  # store the processed corpus
labels = []

x1 = cPickle.load(open("label_ID.p","rb")) # load label 
label_ID = x1[0]
#reading files
file_corpus = open(corpus_path,encoding='utf-8')
for line in file_corpus.readlines():
    [index,content]=line.split('\t')
    messages.append(content)
    if label_ID[index] == 0:
        labels.append(-1)  #nonbullying trace
    else:
        labels.append(1)   #bullying trace



#cleaning content into words
for message in messages:    
    processed_corpus.append(message_to_words(message,del_list))        
cPickle.dump([processed_corpus,labels], open("raw_corpus.p", "wb")) 


#Counting as BoW
vectorizer = CountVectorizer(ngram_range=(1, 2),min_df=2, lowercase=False,tokenizer=lambda doc: doc)   
vec=vectorizer.fit_transform(processed_corpus)   # vec will store the tf-idf model for documents

vocab = vectorizer.get_feature_names()





