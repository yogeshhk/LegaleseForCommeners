#======================================================================================================================
#   Takes list tokens and vectorizes them
#
#   Copyright (C) 2017 Yogesh H Kulkarni
#======================================================================================================================
import numpy as np
from reader import csv_reader, tokenize_text_column

import gensim
from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class
from gensim.corpora import Dictionary
from nltk.corpus import words

LabeledSentence = gensim.models.doc2vec.LabeledSentence  # we'll talk about this down below

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale
saved_dictionary = "./dictionary.dict"

'''
Combine these vectors together and get a new one that represents the tweet as a whole.
A first approach consists in averaging the word vectors together. But a slightly better solution I found was to
compute a weighted average where each weight gives the importance of the word with respect to the corpus.
Such a weight could the tf-idf score. To learn more about tf-idf, you can look at my previous article.
'''
def buildWordVector(tokens, tweet_w2v, tfidf, size = 100):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:  # handling the case where the token is not
            # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

def labelize_docs_dict(docs_dict):
    labelized = []
    for i, v in docs_dict.items():
        label = '%s' % (i)
        tokens = []
        for name, ts in v.items():
            tokens = tokens + ts
        labelized.append(LabeledSentence(tokens, [label]))
    return labelized

def labelize_docs_w_labels(docs, label_type):
    labelized = []
    for i, v in enumerate(docs):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

def vectoriser_models(labelized, size = 100):

    list_tokens = [x.words for x in labelized]

    # build the word2vec model from the corpus.
    w2v_model = Word2Vec(size=size, min_count=1)  # min_count (a threshold for filtering words that appear less)
    w2v_model.build_vocab(list_tokens)
    w2v_model.train(list_tokens)  # its weights are updated.

    print('building tf-idf matrix ...')
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1)
    matrix = vectorizer.fit_transform(list_tokens)
    tfidf_model = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf_model))

    return w2v_model, tfidf_model

def vectorise_column_bow(pandas_tokens_column):
    text_list = np.array(pandas_tokens_column)
    # Makes a list all unique words in all the lists supplied and indexes it like an array, id2word dictionary
    dictionary = Dictionary(text_list)
    additional_documents = []
    additional_documents.append(words.words())
    dictionary.add_documents(additional_documents)
    dictionary.compactify()
    # dictionary.save(saved_dictionary)  # store the dictionary, for future reference
    # print("Dictionary saved at {}".format(saved_dictionary))

    sentenceId_bow_map = {}
    bow_corpus = []
    for i, text in enumerate(text_list):
        bow = dictionary.doc2bow(text)
        bow_corpus.append(bow)
        sentenceId_bow_map[i] = bow
    # # Or we can pass TF IDF-ed corpus as well
    # vectorizer = TfidfVectorizer(min_df=1)
    # corpus = vectorizer.fit_transform(texts)
    return bow_corpus, sentenceId_bow_map, dictionary


def vectorise_column_traintest_tfidfw2v(pandas_tokens_column, pandas_labels_column, tag_train_label, tag_test_label, n_dim =100, split=0.2):
    x_train, x_test, y_train, y_test = train_test_split(np.array(pandas_tokens_column), np.array(pandas_labels_column),test_size=split)

    # Before feeding lists of tokens into the word2vec model, we must turn them into LabeledSentence objects beforehand.
    x_train = labelize_docs_w_labels(x_train, tag_train_label)
    x_test = labelize_docs_w_labels(x_test, tag_test_label)

    w2v_model, tfidf_model = vectoriser_models(x_train)

    train_vecs_w2v = np.concatenate([buildWordVector(z, w2v_model, tfidf_model, n_dim) for z in map(lambda x: x.words, x_train)])
    train_vecs_w2v = scale(train_vecs_w2v)

    test_vecs_w2v = np.concatenate([buildWordVector(z, w2v_model, tfidf_model, n_dim) for z in map(lambda x: x.words, x_test)])
    test_vecs_w2v = scale(test_vecs_w2v)

    return x_train, y_train, train_vecs_w2v, x_test, y_test, test_vecs_w2v, w2v_model, tfidf_model


def vectorise_column_with_tfidfw2v_models(pandas_tokens_column, tag_label, w2v_model, tfidf_model, n_dim):
    unseen_x = np.array(pandas_tokens_column)#(test_df.tokens)
    unseen_x = labelize_docs_w_labels(unseen_x,tag_label)#( 'TEST')
    unseen_list_of_tweets_wordslist = [x.words for x in unseen_x]

    unseen_vecs_w2v = np.concatenate([buildWordVector(z, w2v_model, tfidf_model, n_dim) for z in unseen_list_of_tweets_wordslist])
    unseen_vecs_w2v = scale(unseen_vecs_w2v)

    return unseen_x, unseen_vecs_w2v

if __name__ == "__main__":

    data_csv = '../data/sentiment/SentimentAnalysisDataset.csv'
    columns = ['ItemID', 'label', 'SentimentSource', 'message']
    drops = ['ItemID', 'SentimentSource']
    n_samples = 100

    df = csv_reader(data_csv, columns, drops, n_samples)
    df = tokenize_text_column(df,"message","tokens","tweet")
    df = df[df.label.isnull() == False]
    df['label'] = df['label'].map(int)

    n_dim = 100

    x_train, y_train, train_vecs_w2v, x_test, y_test, test_vecs_w2v, w2v_model, tfidf_model = \
        vectorise_column_traintest_tfidfw2v(df.tokens, df.label, 'TRAIN', 'TEST', n_dim, 0.3)

    print(train_vecs_w2v)
