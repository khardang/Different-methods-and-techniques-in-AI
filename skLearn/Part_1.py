import pickle

import classifier as classifier
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB

data = pickle.load(open("sklearn-data.pickle", "rb"))

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

#n er antall features
#Stop_words er ord som skal fjernes fra teksten, English er en standard liste.
vectorizer = HashingVectorizer(stop_words="english", n_features=2**20, binary=False)

x = vectorizer.fit_transform(x_train)

x_test = vectorizer.fit_transform(x_test)
classifier = BernoulliNB(alpha = 0.1)
classifier.fit(x,y_train)

pred = classifier.predict(x_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy: %0.3f" % score)

