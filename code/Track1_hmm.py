# Ref: https://gist.github.com/blumonkey/007955ec2f67119e0909

import pandas as pd
import os


train_conll_dir = "./data/Task_1/Train_conll/"
test_conll_dir = "./data/Task_1/Test_conll/"


train_data = []
train_docs = os.listdir(train_conll_dir)
for train_doc in train_docs:
    print("Processing {} ...".format(train_doc))
    train_statement_filename = train_conll_dir + train_doc
    train_file_df = pd.read_csv(train_statement_filename, encoding='cp1252', header=None)
    train_file_df.columns = ['token', 'pos', 'iob']
    words = train_file_df['token'].values
    tags = train_file_df['iob'].values
    word_tags = [(w,t) for w,t in zip(words,tags)]
    train_data.append(word_tags)

# Prepare Testing Data df
test_all_df = pd.DataFrame(columns=['token', 'pos'])
test_x = []
test_docs = os.listdir(test_conll_dir)
for test_doc in test_docs:
    print("Processing {} ...".format(test_doc))
    test_statement_filename = test_conll_dir + test_doc
    test_file_df = pd.read_csv(test_statement_filename, encoding='cp1252', header=None)
    test_file_df.columns = ['token', 'pos']
    words = test_file_df['token'].values
    test_x.append(words)

# train_csv = "./data/train.csv"
# test_csv = "./data/test.csv"
#
# train_file_df = pd.read_csv(train_csv, encoding='cp1252', header=None)
# test_file_df = pd.read_csv(test_csv, encoding='cp1252', header=None)
# train_file_df.columns = ['token', 'pos', 'iob']
# test_file_df.columns = ['token', 'pos']
#
# train_x_words = train_file_df['token'].tolist()
# train_y_words = train_file_df['iob'].tolist()
# test_x_words = test_file_df['token'].tolist()
# # print(train_x_words)
# train_data = [(w,t) for w,t in zip(train_x_words,train_y_words)]
# print(train_data)

# Setup a trainer with default(None) values
# And train with the data
from nltk.tag import hmm

print(train_data[0])
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)

print(tagger)
# Prints the basic data about the tagger
for tst in test_x[:3]:
    print(tst)
    test_sentence = " ".join(tst)
    result = tagger.tag(test_sentence.split())
    print(result)

# test_sentence = "defence as pleaded in the written statement and repeating the same in the evidence in chief amounts to contempt of court and convicted the appellant"
# print(tagger.tag(test_sentence.split()))
