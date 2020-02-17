"""
    Generating CONLL like IOB tags from given dataset of court cases and their corresponding catchphrases
    Steps:
        Get every case as cleaned text, split it to form list of words/tokens
        Make a copy of text and replace all words with IoB tags by looking at given catchphrases
        Make another copy of text and replace each word with its POS tag
        Store IOB coded files in Train_conll and Test_conll directories
        Reads all of them and created dataframe, writes those to consolidated csv files
"""
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, word_tokenize
from copy import deepcopy
import csv
import os
import re
import pandas as pd

from collections import OrderedDict

def clean_text(text):
    text = text.replace(".", "") # For words like F.I.R matching becomes a problem, let it come as FIR
    text = text.replace("(", "") # For words like "(something)" matching becomes a problem, let it come as "something"
    text = text.replace(")", "") # For words like "(something)" matching becomes a problem, let it come as "something"
    return text

def clean_statement(text):
    text = clean_text(text)
    tokenizer = RegexpTokenizer(r'\w+')
    wordlist = tokenizer.tokenize(text)
    return " ".join(wordlist)

def replace_in_text(text_to_change,catches):
    iob_str_dict = OrderedDict()
    for catchphrase in catches:
        catchphrase = clean_text(catchphrase)
        words = catchphrase.split()
        iob_str = ' B-LEGAL ' + ' I-LEGAL' * (len(words) - 1)
        iob_str_dict[catchphrase.strip().lower()] = iob_str.strip()

    pattern = re.compile(r'\b(' + '|'.join(iob_str_dict.keys()) + r')\b') # Brackets are for regex groups
    text_changed = pattern.sub(lambda x: iob_str_dict[x.group()], text_to_change)
    return text_changed

def read_statement(filename):
    with open(filename,'r') as r1:
        text = r1.read()
        text = clean_statement(text)
        return text.lower()

def prep_iob_list(iob_text, catches_filename):
    iob_list = []
    with open(catches_filename,'r') as r2:
        catches = r2.read().split(',')
        # Catch phrases with more words should come first
        catches.sort(key = lambda s: len(s.split()),reverse=True)
        # for ct in catches:
        #     iob_text = replace_in_text(iob_text,ct)
        iob_text = replace_in_text(iob_text, catches)

        for word in iob_text.split():
            if word == 'B-LEGAL' or word == 'I-LEGAL':
                iob_list.append(word)
            else:
                iob_list.append('O')
    return iob_list

def prep_pos_list(pos_text):
    pos_list = []
    for w,p in pos_tag(word_tokenize(pos_text)):
        pos_list.append(p)
    return pos_list

def prep_conll_file(iob_filename, text_list, pos_list, iob_list):
    with open(iob_filename, 'w') as f:
        writer = csv.writer(f)
        for t, p, i in zip(text_list, pos_list, iob_list):
            writer.writerow((t,p,i))

def prep_test_file(iob_filename, text_list, pos_list):
    with open(iob_filename, 'w') as f:
        writer = csv.writer(f)
        for t, p in zip(text_list, pos_list):
            writer.writerow((t,p))


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

    # Prepare Training Data
    train_docs = os.listdir(train_statements_dir)
    for train_doc in train_docs:
        if not 'statement' in train_doc:
            continue
        print("Processing {} ...".format(train_doc))
        train_statement_filename = train_statements_dir + train_doc
        train_text = read_statement(train_statement_filename)
        train_text_list = train_text.split()

        train_text_iob = deepcopy(train_text)
        train_catchphrase_filename = train_catchephrases_dir + train_doc.replace('statement', 'catchwords')
        train_list_iob = prep_iob_list(train_text_iob, train_catchphrase_filename)

        train_text_pos = deepcopy(train_text)
        train_list_pos = prep_pos_list(train_text_pos)

        train_conll_filename = train_conll_dir + train_doc.replace('statement', 'conll')
        prep_conll_file(train_conll_filename, train_text_list, train_list_pos, train_list_iob)

    # Prepare Testing data
    test_docs = os.listdir(test_statements_dir)
    for test_doc in test_docs:
        if not 'statement' in test_doc:
            continue
        print("Processing {} ...".format(test_doc))
        test_statement_filename = test_statements_dir + test_doc
        test_text = read_statement(test_statement_filename)
        test_text_list = test_text.split()

        test_text_pos = deepcopy(test_text)
        test_list_pos = prep_pos_list(test_text_pos)

        test_filename = test_conll_dir + test_doc.replace('statement', 'conll')
        prep_test_file(test_filename, test_text_list, test_list_pos)


    # Prepare Training Data df
    train_all_df = pd.DataFrame(columns = ['token', 'pos', 'iob'])
    train_dfs = []
    train_docs = os.listdir(train_conll_dir)
    for train_doc in train_docs:
        print("Processing {} ...".format(train_doc))
        train_statement_filename = train_conll_dir + train_doc
        train_file_df = pd.read_csv(train_statement_filename, encoding='cp1252',header=None)
        print(train_file_df.head())
        train_dfs.append(train_file_df)

    train_all_df = pd.concat(train_dfs, ignore_index=True)
    train_all_df.to_csv(train_csv, encoding='cp1252', index=None, header=None)

    # Prepare Testing Data df
    test_all_df = pd.DataFrame(columns = ['token', 'pos', 'iob'])
    test_dfs = []
    test_docs = os.listdir(test_conll_dir)
    # for test_doc in test_docs: # Making test.csv of only one statement for now.
    test_doc = test_docs[0]
    if test_doc:
        print("Processing {} ...".format(test_doc))
        test_statement_filename = test_conll_dir + test_doc
        test_file_df = pd.read_csv(test_statement_filename, encoding='cp1252',header=None)
        print(test_file_df.head())
        test_dfs.append(test_file_df)

    test_all_df = pd.concat(test_dfs, ignore_index=True)
    test_all_df.to_csv(test_csv, encoding='cp1252', index=None, header=None)