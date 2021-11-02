import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner


class Model:
    def __init__(self, _estimator, _strategy) -> None:
        self.estimator = _estimator
        self.query_strategy = _strategy
        self.learner = None
        self.performance = []
        self.x_test = None
        self.y_test = None
        self.x_pre_train = None
        self.y_pre_train = None
        self.x_pool = None
        self.y_pool = None
        self.x_pre_train_vec = None
        self.x_pool_vec = None
        self.x_test_vec = None

    def init_learner(self, _x_arr, _y_arr, test_size, init_train_size, _vectorizer):
        x, self.x_test, y, self.y_test = train_test_split(_x_arr, _y_arr, test_size=test_size)
        n_elements = len(y)
        pre_train_indices = np.random.randint(low=0, high=n_elements, size=init_train_size)
        self.x_pre_train = x[pre_train_indices]
        self.y_pre_train = y[pre_train_indices]
        self.x_pool = np.delete(x, pre_train_indices)
        self.y_pool = np.delete(y, pre_train_indices)
        _vectorizer.fit(_x_arr)
        self.x_pre_train_vec = _vectorizer.transform(self.x_pre_train)
        self.x_pool_vec = _vectorizer.transform(self.x_pool)
        self.x_test_vec = _vectorizer.transform(self.x_test)
        self.learner = ActiveLearner(
            estimator=self.estimator,
            query_strategy=self.query_strategy,
            X_training=self.x_pre_train_vec,
            y_training=self.y_pre_train
        )

    def score(self):
        score = self.learner.score(self.x_test_vec, self.y_test)
        self.performance.append(score)
        return score

    def query(self):
        idx, _ = self.learner.query(self.x_pool_vec)
        content = self.x_pool[idx][0]
        oracle = self.y_pool[idx][0]
        return {"content": content, "index": idx[0], "oracle": oracle}

    def get_oracle(self, idx):
        return self.y_pool[idx][0]

    def label(self, _idx, _label):
        index = np.array([int(_idx)])
        label = np.array([int(_label)])
        self.learner.teach(X=self.x_pool_vec[index], y=label)

    def auto_run(self, n_round, threshold):
        for i in range(n_round):
            idx, _ = self.learner.query(self.x_pool_vec)
            self.learner.teach(X=self.x_pool_vec[idx], y=self.y_pool[idx])
            score = self.score()
            if score > threshold:
                break
            print(f"round: {i}, scr: {score}")





