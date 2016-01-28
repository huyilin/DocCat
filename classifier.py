__author__ = 'yilinhu'

import cPickle
from  sklearn.feature_extraction.text import CountVectorizer
from  sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import pickle


def get_features():
    f1 = file('training_data.p', 'r')
    data_train = cPickle.load(f1)
    corpus_train = data_train[0]
    labels_train = np.array(data_train[1])
    for i in range(len(corpus_train)):
        corpus_train[i] = ' '.join(corpus_train[i])

    # use counts of words to represent a document
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(corpus_train)

    # apply tf_idf to the vector of counts got above
    tfidf_transformer = TfidfTransformer()
    tfidf_train = tfidf_transformer.fit_transform(matrix)

    f2 = file(r'data/vocab_s_50.p', 'r')
    extended_bad = pickle.load(f2)
    vectorizer_extendted = CountVectorizer().fit(extended_bad[0].keys())
    ebow = vectorizer_extendted.transform(corpus_train)

    return (tfidf_train,)


def get_classifier(features_train, labels_train):
    # create and train a Bayesian classifier using the training data: corpus and labels
    nb_classfier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    nb_classfier = nb_classfier.fit(features_train, labels_train)

    # test the classifier using test documents
    test1_docs = ['good morning, how are you?', 'you are a fucking bitch, idiot, asshole',
                  'Dont bullying your friend', 'I love anna']


# vectorizer = CountVectorizer()
# matrix = vectorizer.fit_transform(corpus_train)
#
# test1_matrix = vectorizer.transform(test1_docs)
# test1_tf_idf = tfidf_transformer.transform(test1_matrix)
# test1_predicted = nb_classfier.predict(test1_tf_idf)
#
# for doc, label in zip(test1_docs, test1_predicted):
#     print doc + '  => ' + str(label)

get_features()
