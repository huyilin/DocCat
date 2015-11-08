__author__ = 'yilinhu'

import cPickle
from  sklearn.feature_extraction .text import CountVectorizer
from  sklearn.feature_extraction .text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# load tranning data, a list of documents(strings) and labels
f = file('trainning_data.p', 'r')
data = cPickle.load(f)
corpus = data[0]
labels = data[1]
for i in range(len(corpus)):
    corpus[i] = ' '.join(corpus[i])
    print corpus[i] + '  => ' + str(labels[i])

# use counts of words to represent a document
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(corpus)

# apply tf_idf to the vector of counts got above
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(counts)

# create and train a Bayesian classifier using the training data: corpus and labels
classfier = MultinomialNB().fit(tfidf, labels)

# test the classifier using test documents
test_docs = ['good morning, how are you?', 'you are a fucking bitch, idiot, asshole',
             'Dont bullying your friend', 'I love anna']
test_counts = vectorizer.transform(test_docs)
test_tf_idf = tfidf_transformer.transform(test_counts)
predicted = classfier.predict(test_tf_idf)

for doc, label in zip(test_docs, predicted):
     print doc + '  => ' + str(label)