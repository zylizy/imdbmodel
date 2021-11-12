import time

from model2 import Model
from sklearn.ensemble import RandomForestClassifier
from modAL.uncertainty import uncertainty_sampling, classifier_entropy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import BernoulliNB
from instance import Instance
import timeit


def save_model():
    _model = Model(RandomForestClassifier(n_classes_=2), uncertainty_sampling)
    data = pd.read_csv("./imdb.csv")
    data['Class'] = np.where(data['Class'].str.contains("P"), 1, 0)
    x_arr = np.array(data["Text"])
    y_arr = np.array(data["Class"])
    tfidf_vectorizer = TfidfVectorizer(binary=True, stop_words='english', ngram_range=(1, 2))
    _model.init_learner(x_arr, y_arr, 0.25, 50, tfidf_vectorizer)
    init_score = _model.score()
    print(init_score)
    with open(r"RF3.pickle", "wb") as outfile:
        pickle.dump(_model, outfile)
    print("Dumped")
    return _model


def load_model(filename, _model):
    with open(f"./{filename}", "rb") as infile:
        _model = pickle.load(infile)
        return _model


def manual_query(_model):

    while True:
        query = _model.query()
        idx = query['index']
        print(f"IDX: {idx}, oracle: {query['oracle']}, content: \n{query['content']}")
        user_in = input("label: (enter \"done\" to terminate)\n")
        if user_in == "done":
            break
        elif user_in not in ['0', '1']:
            print("must be 0,1,done")
            continue
        _model.label(idx, user_in)
        score = _model.score()
        u = _model.learner.predict_proba(_model.x_pool_vec)
        print(score)


if __name__ == "__main__":
    model = None
    # model = save_model()

    model = load_model("RF3.pickle", model)
    #
    a = model.get_proba()


    # queried_instances = {}
    while True:
        print("Waiting for input, type -help to get list of cmds")
        usr_in = input()
        if usr_in == "-help":
            print(f"type q to quit,"
                  f"type a to autorun"
                  f"type m to manually train"
                  f"type i to get all instances queried"
                  f"type u to get uncertainties of all instances queried"
                  f"type ")
        if usr_in == "q":
            break
        elif usr_in == "a":
            model.auto_run(50, 0.7)
        elif usr_in == "m":
            manual_query(model)
        elif usr_in == "i":
            print(queried_instances)
        elif usr_in == "u":
            print("?")




