
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from modAL.models import ActiveLearner
# from modAL.uncertainty import uncertainty_sampling, classifier_entropy
# # from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier
# import pickle

# class ALModel:
#
#     def __init__(self, path, estimator) -> None:
#         self.data_path = path
#         self.estimator = estimator
#         self.sample_strategy = classifier_entropy
#         self.learner = None
#         self.performance = []
#         self.X_train = None
#         self.Y_train = None
#         self.X_pool = None
#         self.Y_pool = None
#         self.tfidf_vectorizer = None
#         self.X_train_tfidf = None
#         self.X_pool_tfidf = None
#         self.X_test_tfidf = None
#         self.Y_test = None
#
#     def query(self):
#         idx, _ = self.learner.query(self.X_pool_tfidf)
#         return {
#             "content": self.X_pool[idx][0],
#             "idx": idx[0],
#             "oracle": self.Y_pool[idx][0]
#         }
#
#     def label(self, _label, _idx):
#         index = np.array([int(_idx)])
#         label = np.array([int(_label)])
#         self.learner.teach(X=self.X_pool_tfidf[index], y=label)
#
#         return self.score()
#
#     def score(self):
#         score = self.learner.score(self.X_test_tfidf, self.Y_test)
#         self.performance.append(score)
#         return float(score)
#
#     def pretraining(self):
#         data = pd.read_csv(self.data_path)
#         data['Class'] = np.where(data['Class'].str.contains("P"), 1, 0)
#         X_raw = np.array(data['Text'])
#         Y_raw = np.array(data['Class'])
#         X, X_test, Y, self.Y_test = train_test_split(X_raw, Y_raw, test_size=0.25)
#         n_elements = len(Y)
#         training_indices = np.random.randint(low=0, high=n_elements, size=30)
#         self.X_train = X[training_indices]
#         self.Y_train = Y[training_indices]
#         self.X_pool = np.delete(X, training_indices)
#         self.Y_pool = np.delete(Y, training_indices)
#         self.tfidf_vectorizer = TfidfVectorizer(binary=True, stop_words='english', ngram_range=(1, 2))
#         self.tfidf_vectorizer.fit(X)
#         self.X_train_tfidf = self.tfidf_vectorizer.transform(self.X_train)
#         self.X_pool_tfidf = self.tfidf_vectorizer.transform(self.X_pool)
#         self.X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
#         self.learner = ActiveLearner(
#             estimator=self.estimator,
#             query_strategy=self.sample_strategy,
#             X_training=self.X_train_tfidf,
#             y_training=self.Y_train)
#         return self.score()
#
#     def get_performance(self):
#         return self.performance
#
#     def save_learner(self):
#         pickle.dump(self.learner, open("learner.pickle", "wb"))

from math import e
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.naive_bayes import MultinomialNB


class ALModel:

    def __init__(self, path, estimator) -> None:
        self.data_path = path
        self.estimator = estimator
        self.sample_strategy = uncertainty_sampling
        self.learner = None
        self.performance = []
        self.X_train = None
        self.Y_train = None
        self.X_pool = None
        self.Y_pool = None
        self.tfidf_vectorizer = None
        self.X_train_tfidf = None
        self.X_pool_tfidf = None
        self.X_test_tfidf = None
        self.Y_test = None

    def query(self):
        idx, _ = self.learner.query(self.X_pool_tfidf)
        return {
            "content": self.X_pool[idx][0],
            "idx": idx[0],
            "oracle": self.Y_pool[idx][0]
        }

    def label(self, _label, _idx):
        index = np.array([int(_idx)])
        label = np.array([int(_label)])
        self.learner.teach(X=self.X_pool_tfidf[index], y=label)

        return self.score()

    def score(self):
        score = self.learner.score(self.X_test_tfidf, self.Y_test)
        self.performance.append(score)
        return float(score)

    def pretraining(self):
        data = pd.read_csv(self.data_path)
        data['Class'] = np.where(data['Class'].str.contains("P"), 1, 0)
        X = np.array(data['Text'])
        Y = np.array(data['Class'])
        X, X_test, Y, self.Y_test = train_test_split(X, Y, test_size=0.25)
        n_elements = len(Y)
        training_indices = np.random.randint(low=0, high=n_elements, size=30)
        self.X_train = X[training_indices]
        self.Y_train = Y[training_indices]
        self.X_pool = np.delete(X, training_indices)
        self.Y_pool = np.delete(Y, training_indices)
        self.tfidf_vectorizer = TfidfVectorizer(binary=True, stop_words='english', ngram_range=(1, 2))
        self.tfidf_vectorizer.fit(X)
        self.X_train_tfidf = self.tfidf_vectorizer.transform(self.X_train)
        self.X_pool_tfidf = self.tfidf_vectorizer.transform(self.X_pool)
        self.X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        self.learner = ActiveLearner(
            estimator=self.estimator,
            query_strategy=self.sample_strategy,
            X_training=self.X_train_tfidf,
            y_training=self.Y_train)
        self.score()

    def get_performance(self):
        return self.performance

