# Ref: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# Problem post: https://groups.google.com/forum/#!topic/keras-users/wMcsmaAUTBY

# Keras==2.0.2
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, word_tokenize
from copy import deepcopy
import csv
import os
import re
import pandas as pd

from collections import OrderedDict


if __name__ == "__main__":
    train_statements_dir = "./data/Task_1/Train_docs/"
    train_catchephrases_dir = "./data/Task_1/Train_catches/"
    train_conll_dir = "./data/Task_1/Train_conll/"

    test_statements_dir = "./data/Task_1/Test_docs/"
    test_conll_dir = "./data/Task_1/Test_conll/"

    train_csv = "./data/train.csv"
    test_csv = "./data/test.csv"

    # Just for debugging few local files, uncomment following
    # statements_dir = "./data/"
    # catches_dir = "./data/"
    # conll_dir = "./data/"

    # Prepare Training Data df
    train_all_df = pd.DataFrame(columns = ['token', 'pos', 'iob'])
    X = []
    y = []
    train_docs = os.listdir(train_conll_dir)
    for train_doc in train_docs:
        print("Processing {} ...".format(train_doc))
        train_statement_filename = train_conll_dir + train_doc
        train_file_df = pd.read_csv(train_statement_filename, encoding='cp1252',header=None)
        train_file_df.columns = ['token', 'pos', 'iob']
        words = train_file_df['token'].values
        tags = train_file_df['iob'].values
        if len(words) < 1000: ########## REMOVE THIS LATER
            X.append(words)
            y.append(tags)

    # Prepare Testing Data df
    test_all_df = pd.DataFrame(columns = ['token', 'pos'])
    test_x = []
    test_docs = os.listdir(test_conll_dir)
    for test_doc in test_docs:
        print("Processing {} ...".format(test_doc))
        test_statement_filename = test_conll_dir + test_doc
        test_file_df = pd.read_csv(test_statement_filename, encoding='cp1252',header=None)
        test_file_df.columns = ['token', 'pos']
        words = test_file_df['token'].values
        if len(words) < 1000: ########## REMOVE THIS LATER
            test_x.append(words)

    #
    # raw = open('./data/wikigold.conll.txt', 'r', encoding='utf-8').readlines()
    #
    # all_x = []
    # point = []
    # for line in raw:
    #     stripped_line = line.strip().split(' ')
    #     point.append(stripped_line)
    #     if line == '\n':
    #         all_x.append(point[:-1])
    #         point = []
    # all_x = all_x[:-1]
    #
    # lengths = [len(x) for x in all_x]
    # print('Input sequence length range: {} {} '.format(max(lengths), min(lengths)))
    #
    # short_x = [x for x in all_x if len(x) < 64]
    #
    # X = [[c[0] for c in x] for x in short_x]
    # y = [[c[1] for c in y] for y in short_x]

    all_text = [c for x in X for c in x]
    all_text = all_text + [c for x in test_x for c in x]

    words = list(set(all_text))
    word2ind = {word: index for index, word in enumerate(words)}
    ind2word = {index: word for index, word in enumerate(words)}
    labels = list(set([c for x in y for c in x]))
    label2ind = {label: (index + 1) for index, label in enumerate(labels)}
    ind2label = {(index + 1): label for index, label in enumerate(labels)}
    print('Vocabulary size: {} {}'.format(len(word2ind), len(label2ind)))

    maxlen = max([len(x) for x in X])
    print('Maximum sequence length: {}'.format(maxlen))


    def encode(x, n):
        result = np.zeros(n)
        result[x] = 1
        return result


    X_enc = [[word2ind[c] for c in x] for x in X]
    max_label = max(label2ind.values()) + 1 # The extra one is for the PAD I guess
    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

    X_enc = pad_sequences(X_enc, maxlen=maxlen)
    y_enc = pad_sequences(y_enc, maxlen=maxlen)
    n_samples = len(X_enc)

    # X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=11 * 32, train_size=45 * 32,random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=int(0.3*n_samples), train_size=int(0.7*n_samples),random_state=42)
    print('Training and testing tensor shapes: {} {} {} {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    max_features = len(word2ind)
    embedding_size = 128
    hidden_size = 32
    out_size = len(label2ind) + 1

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True, go_backwards=True))
    model.add(TimeDistributed(Dense(out_size)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    batch_size = 32
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=50, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Raw test score:', score)


    def score(yh, pr):
        coords = [np.where(yhh > 0)[0][0] for yhh in yh]
        yh = [yhh[co:] for yhh, co in zip(yh, coords)]
        ypr = [prr[co:] for prr, co in zip(pr, coords)]
        fyh = [c for row in yh for c in row]
        fpr = [c for row in ypr for c in row]
        return fyh, fpr


    pr = model.predict_classes(X_train)
    yh = y_train.argmax(2)
    fyh, fpr = score(yh, pr)
    print('Training accuracy: {}'.format(accuracy_score(fyh, fpr)))
    print('Training confusion matrix: {}'.format(confusion_matrix(fyh, fpr)))
    precision_recall_fscore_support(fyh, fpr)

    pr = model.predict_classes(X_test)
    yh = y_test.argmax(2)
    fyh, fpr = score(yh, pr)
    print('Training accuracy: {}'.format(accuracy_score(fyh, fpr)))
    print('Training confusion matrix: {}'.format(confusion_matrix(fyh, fpr)))
    precision_recall_fscore_support(fyh, fpr)

    # Testing
    test_x_enc = [[word2ind[c] for c in x] for x in test_x]
    test_x_enc = pad_sequences(test_x_enc, maxlen=maxlen)
    pr = model.predict_classes(np.array(test_x_enc))
    tags = [[ind2label[c] for c in x] for x in pr]
    words = [[ind2word[c] for c in x] for x in test_x_enc]

    print(tags)
    prediction_df = pd.DataFrame(columns=['token', 'iob'])
    all_tokens = []
    all_tags = []
    for token_list, tag_list in zip(words,tags):
        all_tokens = all_tokens + token_list
        all_tags = all_tags + tag_list
    prediction_df['token'] = all_tokens
    prediction_df['iob'] = all_tags
    prediction_df = prediction_df[prediction_df.token != 0]
    prediction_df.to_csv('./data/newpredictions.csv', index=None,header=None)


