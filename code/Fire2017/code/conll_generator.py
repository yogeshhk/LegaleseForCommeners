"""
    Experiment: Generating CONLL like IOB tags from given dataset of court cases and their corresponding catchphrases
    Steps:
        Get every case as cleaned text, split it to form list of words/tokens
        Make a copy of text and replace all words with IoB tags by looking at given catchphrases
        Make another copy of text and replace each word with its POS tag
        Store IOB coded files in Train_conll directory
"""
from nltk.tokenize import RegexpTokenizer
from copy import deepcopy
import csv
import nltk
import os
import re
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
    print(iob_str_dict)

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
        iob_text = replace_in_text(iob_text, catches)
        for word in iob_text.split():
            if word == 'B-LEGAL' or word == 'I-LEGAL':
                iob_list.append(word)
            else:
                iob_list.append('O')
    return iob_list

def prep_pos_list(pos_text):
    pos_list = []
    for w,p in nltk.pos_tag(pos_text):
        pos_list.append(p)
    return pos_list

def prep_iob_file(iob_filename,text_list,pos_list,iob_list):
    with open(iob_filename, 'w') as f:
        writer = csv.writer(f)
        for t, p, i in zip(text_list, pos_list, iob_list):
            writer.writerow((t,p,i))

if __name__ == "__main__":
    statements_dir = "../Task_1/Train_docs/"
    catches_dir = "../Task_1/Train_catches/"
    conll_dir = "../Task_1/Train_conll/"

    # Just for debugging few local files, uncomment following
    # statements_dir = "./data/"
    # catches_dir = "./data/"
    # conll_dir = "./data/"

    docs = os.listdir(statements_dir)
    for doc in docs:
        if not 'statement' in doc:
            continue
        print("Processing {} ...".format(doc))
        statement_filename = statements_dir + doc
        text = read_statement(statement_filename)
        text_list = text.split()

        iob_text = deepcopy(text)
        catches_filename = catches_dir + doc.replace('statement','catchwords')
        iob_list = prep_iob_list(iob_text,catches_filename)

        pos_text = deepcopy(text)
        pos_list = prep_pos_list(pos_text)

        iob_filename = conll_dir + doc.replace('statement','conll')
        prep_iob_file(iob_filename,text_list,pos_list,iob_list)