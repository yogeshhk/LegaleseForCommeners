# Ref: https://spacy.io/docs/usage/training-ner
# Problem post: https://github.com/explosion/spaCy/issues/665

import csv
import os
import re
import pandas as pd

from collections import OrderedDict

import spacy
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
from spacy.tagger import Tagger
import random
from spacy.gold import _iob_to_biluo as iob_to_biluo

model_name = 'en'
entity_label = 'LEGAL'
output_directory = 'C:/Users/kulkarni/Downloads/SpacyModel'

def train_ner(nlp, train_data, output_dir):
    for raw_text, _ in train_data:
        doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]

    for itn in range(20):
        random.shuffle(train_data)
        for raw_text, entity_tags in train_data:
            try:
                doc = nlp.make_doc(raw_text)
                gold = GoldParse(doc, entities=entity_tags)
            except IndexError:
                # print("Doc {}".format(doc))
                # print("Tags {}".format(entity_tags))
                print("Got error for...tags present {}".format("B-LEGAL" in entity_tags))
            else:
                nlp.tagger(doc)
                loss = nlp.entity.update(doc, gold)
    nlp.end_training()
    nlp.save_to_directory(output_dir)

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
        # tags = [w.replace('B-LEGAL', "U-" + entity_label) for w in tags] #U- is needed for tags, but add_label dosen't
        # tags = [w.replace('I-LEGAL', "U-" + entity_label) for w in tags]
        # tags = iob_to_biluo(tags)
        # if len(words) < 1000: ########## REMOVE THIS LATER
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
        # if len(words) < 1000: ########## REMOVE THIS LATER
        test_x.append(words)

    nlp = spacy.load(model_name)
    nlp.entity.add_label(entity_label)
    train_data = [(" ".join(text), tags) for text, tags in zip(X,y)]
    # train_data = [("The fact of the matter is that the case is open", ['O','O','O','O','O','O','O','B-LEGAL','I-LEGAL','O','O']),
    #               ("It is not decided that the case should be shut", ['O','O','O','O','O','B-LEGAL','I-LEGAL','O','O','O']),
    #               ("Why not look at the case closely",['O','O','O','O','B-LEGAL','I-LEGAL','O'])]
    ner = train_ner(nlp, train_data, output_directory)
    # test_x = ["Please open the case whenever possible","Its i mandatory to examine the case"]
    for words in test_x:
        raw_text = " ".join(words)
        # raw_text = words
        doc = nlp.make_doc(raw_text)
        nlp.entity(doc)
        for ent in doc.ents:
            print(ent.label_, ent.text)
        # GPE London
        # GPE United Kingdom