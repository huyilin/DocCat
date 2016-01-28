__author__ = 'yilinhu'

import cPickle
from  sklearn.feature_extraction .text import CountVectorizer
from  sklearn.feature_extraction .text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

# load tranning data, a list of documents(strings) and labels
##############################
# An example of classification
target_names = ['bulling trace', 'nonbullying trace']
f = file('training_data.p', 'r')
data_train = cPickle.load(f)
corpus_train = data_train[0]
labels_train = np.array(data_train[1])
for i in range(len(corpus_train)):
    corpus_train[i] = ' '.join(corpus_train[i])
    # print corpus[i] + '  => ' + str(labels[i])

# use counts of words to represent a document
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(corpus_train)

# apply tf_idf to the vector of counts got above
tfidf_transformer = TfidfTransformer()
tfidf_train = tfidf_transformer.fit_transform(matrix)

# create and train a Bayesian classifier using the training data: corpus and labels
nb_classfier = MultinomialNB()
nb_classfier = nb_classfier.fit(tfidf_train, labels_train)

# test the classifier using test documents
test1_docs = ['good morning, how are you?', 'you are a fucking bitch, idiot, asshole',
             'Dont bullying your friend', 'I love anna']
test1_matrix = vectorizer.transform(test1_docs)
test1_tf_idf = tfidf_transformer.transform(test1_matrix)
test1_predicted = nb_classfier.predict(test1_tf_idf)

for doc, label in zip(test1_docs, test1_predicted):
      print doc + '  => ' + str(label)
##############################
# loading test data

f = file('test_data.p', 'r')
data_test = cPickle.load(f)
corpus_test = data_test[0]
labels_test = np.array(data_test[1])
for i in range(len(corpus_test)):
    corpus_test[i] = ' '.join(corpus_test[i])

# make the vectorizer => transformer => classifier easier to work with,
# scikit-learn provides a Pipeline class that behaves like a compound classifier:

# try first with naive bayes classifier

text_classifier = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultinomialNB()),])

text_classifier = text_classifier.fit(corpus_train, labels_train)
predicted_test = text_classifier.predict(corpus_test)
performance = metrics.classification_report(labels_test, predicted_test, [-1 , 1])
print "performance of using Naive Bayes"
print performance


# Try using the Support Vector Machine
svm_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),
])

svm_clf = svm_clf.fit(corpus_train, labels_train)
predicted_test = svm_clf.predict(corpus_test)
performance = metrics.classification_report(labels_test, predicted_test, [-1 , 1])
print "performance of using Support Vector Machine"
print performance