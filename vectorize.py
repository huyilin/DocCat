__author__ = 'yilinhu'

import cPickle
from  sklearn.feature_extraction .text import CountVectorizer
from  sklearn.feature_extraction .text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline

# load tranning data, a list of documents(strings) and labels
f = file('trainning_data.p', 'r')
data_train = cPickle.load(f)
corpus_train = data_train[0]
labels_train = data_train[1]
for i in range(len(corpus_train)):
    corpus_train[i] = ' '.join(corpus_train[i])
    # print corpus[i] + '  => ' + str(labels[i])

# use counts of words to represent a document
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(corpus_train)

# apply tf_idf to the vector of counts got above
tfidf_transformer = TfidfTransformer()
tfidf_train = tfidf_transformer.fit_transform(counts)

# create and train a Bayesian classifier using the training data: corpus and labels
classfier = MultinomialNB().fit(tfidf_train, labels_train)

# test the classifier using test documents
test1_docs = ['good morning, how are you?', 'you are a fucking bitch, idiot, asshole',
             'Dont bullying your friend', 'I love anna']
test1_counts = vectorizer.transform(test1_docs)
test1_tf_idf = tfidf_transformer.transform(test1_counts)
test1_predicted = classfier.predict(test1_tf_idf)

for doc, label in zip(test1_docs, test1_predicted):
     print doc + '  => ' + str(label)



# make the vectorizer => transformer => classifier easier to work with,
# scikit-learn provides a Pipeline class that behaves like a compound classifier:

text_classifier = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultinomialNB()),])

text_classifier = text_classifier.fit(corpus_train, labels_train)

# see how does it perform for trainning data
predicted_train = text_classifier.predict(corpus_train)
print 'accuracy for trainning data : ' + str(np.mean(predicted_train == labels_train))

# using classifer to test the test data and get precision rate
f = file('test_data.p', 'r')
data_test = cPickle.load(f)
corpus_test = data_test[0]
labels_test = data_test[1]
for i in range(len(corpus_test)):
    corpus_test[i] = ' '.join(corpus_test[i])

predicted_train = text_classifier.predict(corpus_train)
predicted_test = text_classifier.predict(corpus_test)
print 'accuracy for test data : ' + str(np.mean(predicted_test == labels_test))

# Try using the Support Vector Machine