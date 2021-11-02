import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import pickle
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner
from sklearn.naive_bayes import BernoulliNB
from modAL.uncertainty import entropy_sampling
# from modAL.uncertainty import uncertainty_sampling, classifier_entropy


def get_stopwords():
    with open("stopwords.txt") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        return lines


def preprocessor(doc):
    sw = get_stopwords()
    tokenizer = RegexpTokenizer(r'\w+')
    stems = []
    tokens = [token for token in tokenizer.tokenize(doc) if token not in sw and token.isalpha()]
    for token in tokens:
        stems.append(SnowballStemmer("english", ignore_stopwords=True).stem(token))
    return " ".join(stems)


# def version1(_filename):
#     df = pd.read_csv(f"./{_filename}")
#     # 1 for positive and 0 for negative
#     df['Class'] = np.where(df['Class'].str.contains("P"), 1, 0)
#     reviews = np.array(df['Text'])
#     vectorizer = TfidfVectorizer(preprocessor=preprocessor, binary=True, ngram_range=(1, 2))
#     vectorizer.fit(reviews)
#     pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
#     pickle.dump(df, open("imdb_df.pickle", "wb"))


def save_df_vectorizer(_filename):
    df = pd.read_csv(f"./{_filename}")
    # 1 for positive and 0 for negative
    df['Class'] = np.where(df['Class'].str.contains("P"), 1, 0)
    reviews = np.array(df['Text'])
    labels = np.array(df['Class'])
    reviews_raw, reviews_raw_test, labels_raw, labels_raw_test = train_test_split(reviews, labels, test_size=0.25)
    training_indices = np.random.randint(low=0, high=len(labels_raw), size=30)
    reviews_train = reviews_raw[training_indices]
    labels_train = labels_raw[training_indices]
    reviews_pool = np.delete(reviews_raw, training_indices)
    labels_pool = np.delete(labels_raw, training_indices)
    tfidf_vectorizer = TfidfVectorizer(binary=True, stop_words='english', ngram_range=(1, 2))
    tfidf_vectorizer.fit(reviews)

    reviews_train_tfidf = tfidf_vectorizer.transform(reviews_train)
    reviews_pool_tfidf = tfidf_vectorizer.transform(reviews_pool)
    reviews_raw_test_tfidf = tfidf_vectorizer.transform(reviews_raw_test)

    learner = ActiveLearner(
        estimator=BernoulliNB(),
        query_strategy=entropy_sampling,
        X_training=reviews_train_tfidf,
        y_training=labels_train)
    init_score = learner.score(reviews_raw_test_tfidf, labels_raw_test)
    print(init_score)

    # -------------------------------
    while True:
        idx_, _ = learner.query(reviews_pool_tfidf)

        content = reviews_pool[idx_][0]
        idx = idx_[0]
        oracle = labels_pool[idx]
        print(f"IDX: {idx}, oracle: {oracle}, content: \n{content}")
        user_in = input("label: (enter \"done\" to terminate)\n")
        if user_in == "done":
            break
        elif user_in not in ['0', '1']:
            print("must be 0,1,done")
            continue

        index = np.array([int(idx)])
        label = np.array([int(user_in)])
        learner.teach(X=reviews_pool_tfidf[index], y=label)

        score = learner.score(reviews_raw_test_tfidf, labels_raw_test)
        print(score)


if __name__ == "__main__":
    df = pd.read_csv("imdb.csv")
    # 1 for positive and 0 for negative
    df['Class'] = np.where(df['Class'].str.contains("P"), 1, 0)
    reviews = np.array(df['Text'])
    labels = np.array(df['Class'])
    reviews_raw, reviews_raw_test, labels_raw, labels_raw_test = train_test_split(reviews, labels, test_size=0.25)
    training_indices = np.random.randint(low=0, high=len(labels_raw), size=30)
    reviews_train = reviews_raw[training_indices]
    labels_train = labels_raw[training_indices]
    reviews_pool = np.delete(reviews_raw, training_indices)
    labels_pool = np.delete(labels_raw, training_indices)
    tfidf_vectorizer = TfidfVectorizer(binary=True, stop_words='english', ngram_range=(1, 2))
    tfidf_vectorizer.fit(reviews)

    reviews_train_tfidf = tfidf_vectorizer.transform(reviews_train)
    reviews_pool_tfidf = tfidf_vectorizer.transform(reviews_pool)
    reviews_raw_test_tfidf = tfidf_vectorizer.transform(reviews_raw_test)

    learner = ActiveLearner(
        estimator=BernoulliNB(),
        query_strategy=entropy_sampling,
        X_training=reviews_train_tfidf,
        y_training=labels_train)
    init_score = learner.score(reviews_raw_test_tfidf, labels_raw_test)
    print(init_score)


    while True:
        idx_, _ = learner.query(reviews_pool_tfidf)

        content = reviews_pool[idx_][0]
        idx = idx_[0]
        oracle = labels_pool[idx]
        print(f"IDX: {idx}, oracle: {oracle}, content: \n{content}")
        user_in = input("label: (enter \"done\" to terminate)\n")
        if user_in == "done":
            break
        elif user_in not in ['0', '1']:
            print("must be 0,1,done")
            continue

        index = np.array([int(idx)])
        label = np.array([int(user_in)])
        learner.teach(X=reviews_pool_tfidf[index], y=label)

        score = learner.score(reviews_raw_test_tfidf, labels_raw_test)
        print(score)

