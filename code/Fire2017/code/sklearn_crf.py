import pandas as pd
import pycrfsuite


if __name__ == "__main__":
    train_csv = "./data/train.csv"
    test_csv = "./data/test.csv"

    train_df = pd.read_csv(train_csv, encoding='cp1252', header=None)
    test_df = pd.read_csv(test_csv, encoding='cp1252', header=None)


    # trainer = pycrfsuite.Trainer(verbose=False)
    #
    # trainer.set_params({
    #     'c1': 1.0,  # coefficient for L1 penalty
    #     'c2': 1e-3,  # coefficient for L2 penalty
    #     'max_iterations': 50,  # stop earlier
    #
    #     # include transitions that are possible, but not observed
    #     'feature.possible_transitions': True
    # })
    #
    # for index, row in train_df.iterrows():
    #     xseq = (row[0],row[1])
    #     yseq = row[2]
    #     trainer.append(xseq, yseq)
    #
    # pickle_file = ',/data/conll_crfsuite'
    # trainer.train(pickle_file)
    #
    # tagger = pycrfsuite.Tagger()
    # tagger.open(pickle_file)
    #
    # for index, row in test_df.iterrows():
    #     xseq = (row[0],row[1])
    #     tag = tagger.tag(xseq)
    #     print(tag)