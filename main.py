from model import ALModel
from sklearn.ensemble import RandomForestClassifier
import pickle


if __name__ == "__main__":
    learner = ALModel("./imdb.csv", RandomForestClassifier())
    init_score = learner.pretraining()
    print(init_score)
    with open(r"learner.pickle", "wb") as outfile:
        pickle.dump(learner, outfile)
    print("Dumped")
    learner = None
    with open(r"learner.pickle", "rb") as infile:
        learner = pickle.load(infile)
    print("load")

    while True:
        query = learner.query()
        idx = query['idx']
        print(f"IDX: {idx}, oracle: {query['oracle']}, content: \n{query['content']}")
        user_in = input("label: (enter \"done\" to terminate)\n")
        if user_in == "done":
            break
        elif user_in not in ['0', '1']:
            print("must be 0,1,done")
            continue
        score = learner.label(user_in, idx)
        print(score)
