#======================================================================================================================
#   Using WordVectors in SVM for classification
#
#   Reference:
#       http://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html
#   Copyright (C) 2017 Yogesh H Kulkarni
#======================================================================================================================

import pandas as pd
pd.options.mode.chained_assignment = None

from reader import csv_reader, tokenize_text_column
from vectoriser import vectorise_column_traintest_tfidfw2v, vectorise_column_with_tfidfw2v_models

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def prep_for_training(data_csv, columns, drops, n_dim, n_samples):
    train_df = csv_reader(data_csv, columns, drops, n_samples)
    train_df = tokenize_text_column(train_df,"message","tokens","tweet")
    train_df = train_df[train_df.label.isnull() == False]
    train_df['label'] = train_df['label'].map(int)

    x_train, y_train, train_vecs_w2v, x_test, y_test, test_vecs_w2v, w2v_model, tfidf_model = \
        vectorise_column_traintest_tfidfw2v(train_df.tokens, train_df.label, 'TRAIN', 'TEST', n_dim)

    return x_train, y_train, train_vecs_w2v, x_test, y_test, test_vecs_w2v, w2v_model, tfidf_model


def prepare_for_testing(test_data_csv, columns, drops, w2v_model, tfidf_model, n_dim, n_samples):
    test_df = csv_reader(test_data_csv, columns, drops, n_samples)
    test_df = tokenize_text_column(test_df,"message","tokens","tweet")
    test_df.reset_index(inplace=True)
    test_df.drop('index', inplace=True, axis=1)
    unseen_x, unseen_vecs_w2v =\
        vectorise_column_with_tfidfw2v_models(test_df.tokens, 'TEST', w2v_model, tfidf_model, n_dim)

    return unseen_x, unseen_vecs_w2v


if __name__ == "__main__":

    n_dim = 100

    # Training
    train_data_csv = '../data/sentiment/SentimentAnalysisDataset.csv'
    train_columns = ['ItemID', 'label', 'SentimentSource', 'message']
    drop_columns = ['ItemID', 'SentimentSource']
    n_samples = 50000
    x_train, y_train, train_vecs_w2v, x_test, y_test, test_vecs_w2v, w2v_model, tfidf_model = \
        prep_for_training(train_data_csv, train_columns, drop_columns, n_dim, n_samples)

    # Modeling
    print('Running classifier...')
    classifier_model = SVC().fit(train_vecs_w2v, y_train)
    y_test_pred = classifier_model.predict(test_vecs_w2v)
    score = accuracy_score(y_test, y_test_pred.astype(int))
    print(score)


    # Testing
    test_data_csv = '../data/chatmessages/sample_data.csv'
    test_columns = ['ConversationId', 'PersonId', 'message','Date']
    drop_columns = ['ConversationId', 'PersonId', 'Date']
    n_samples = 100
    unseen_x, unseen_vecs_w2v = prepare_for_testing(test_data_csv, test_columns, drop_columns, w2v_model, tfidf_model, n_dim, n_samples)

    y_unseen_pred = classifier_model.predict(unseen_vecs_w2v)
    for i, j in zip(unseen_x, y_unseen_pred):
        message = " ".join(i[0])
        prediction = "Negative"
        if j == 1:
            prediction = "Positive"
        print("\"{}\" :\t{}".format(message, prediction))
