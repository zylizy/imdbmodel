from model2 import Model
from sklearn.ensemble import RandomForestClassifier
from modAL.uncertainty import uncertainty_sampling, classifier_entropy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import BernoulliNB


if __name__ == "__main__":
    # model = Model(BernoulliNB(), uncertainty_sampling)
    # data = pd.read_csv("./imdb.csv")
    # data['Class'] = np.where(data['Class'].str.contains("P"), 1, 0)
    # x_arr = np.array(data["Text"])
    # y_arr = np.array(data["Class"])
    # tfidf_vectorizer = TfidfVectorizer(binary=True, stop_words='english', ngram_range=(1, 2))
    # model.init_learner(x_arr, y_arr, 0.25, 50, tfidf_vectorizer)
    # init_score = model.score()
    # print(init_score)
    #
    # with open(r"BNmodel.pickle", "wb") as outfile:
    #     pickle.dump(model, outfile)
    # print("Dumped")
    #
    model = None
    with open(r"BNmodel.pickle", "rb") as infile:
        model = pickle.load(infile)
    print("load")
    model.auto_run(500, 0.8)



    #
    # while True:
    #     query = model.query()
    #     idx = query['index']
    #     print(f"IDX: {idx}, oracle: {query['oracle']}, content: \n{query['content']}")
    #     user_in = input("label: (enter \"done\" to terminate)\n")
    #     if user_in == "done":
    #         break
    #     elif user_in not in ['0', '1']:
    #         print("must be 0,1,done")
    #         continue
    #     model.label(user_in, idx)
    #     score = model.score()
    #     print(score)
