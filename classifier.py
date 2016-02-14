__author__ = 'anna'

import cPickle
from  sklearn.feature_extraction.text import CountVectorizer
from  sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from numpy import concatenate
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import metrics
import pickle
from sklearn.pipeline import Pipeline
import scipy


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
    extended_bad = cPickle.load(f2)
    bword_train = extended_bad[0]
    weight_train = np.array(extended_bad[1])
    vectorizer_extendted = CountVectorizer().fit(extended_bad[0].keys())
    ebow_train = vectorizer_extendted.transform(corpus_train)
    # for i in range(len(weight_train)):
    # ebow_train[i] = ebow_train[i] * weight_train[i]

    svd_transformer = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

    svd_train = svd_transformer.fit_transform(matrix)

    # features_train = np.concatenate((tfidf_train,ebow_train), axis=0)
    features_train = scipy.sparse.hstack((tfidf_train, ebow_train, svd_train), format='csr')
    return (features_train, labels_train, vectorizer, tfidf_transformer, vectorizer_extendted, svd_transformer)


def get_classifier(features_train, labels_train):
    # create and train a Bayesian classifier using the training data: corpus and labels
    nb_classfier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    nb_classfier = nb_classfier.fit(features_train, labels_train)
    # svm_classifier = svm.svc()
    # classifier_svm = svm_classifier.fit(features_train,labels_train)

    return (nb_classfier)

    # svm_classifier = svm.svc()
    # classifier_svm = svm_classifier.fit(features_train,labels_train)


features_train, labels_train, vectorizer, tfidf_transformer, vectorizer_extendted, svd_transformer = get_features()
nb_classifier = get_classifier(features_train, labels_train)

# test the classifier using test documents
test1_docs = ['good morning, how are you?', 'you are a fucking bitch, idiot, asshole',
              'Dont bullying your friend', 'I love anna']

test1_matrix = vectorizer.transform(test1_docs)
test1_tf_idf = tfidf_transformer.transform(test1_matrix)
test1_svd = svd_transformer.transform(test1_matrix)
test1_ebow = vectorizer_extendted.transform(test1_docs)
test1_features = scipy.sparse.hstack((test1_tf_idf, test1_ebow, test1_svd), format='csr')

test1_predicted = nb_classifier.predict(test1_features)

for doc, label in zip(test1_docs, test1_predicted):
    print doc + '  => ' + str(label)
