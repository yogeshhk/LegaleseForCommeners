#======================================================================================================================
#   Reads many types of sources into pandas dataframe, with min two columns, doc_name, text
#
#   References:
#       https://raw.githubusercontent.com/skcript/cvscan/master/cvscan/converter.py
#   Copyright (C) 2017 Yogesh H Kulkarni
#======================================================================================================================

import re
import os

import pandas as pd  # provide sql-like data manipulation tools. very handy.
from preprocessor import clean_and_tokenize_messages, clean_and_tokenize_sentences
"""
OCR teseract needs a tiff file, needs core tesseract be installed from
https://github.com/UB-Mannheim/tesseract/wiki

"""
from PIL import Image as pilImage
import pytesseract
import os
def to_text_by_tesseract(path):
    return pytesseract.image_to_string(pilImage.open(path))


"""
Utility Function to convert pdfs to plain txt format.
Derived from the examples provided by the pdfminer package documentation.
Params: file_name type: string
returns string
"""
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
# from cStringIO import StringIO
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def to_text_by_pdfminer(path):
    "Wrapper around pdfminer."
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    laparams.all_texts = True
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    pages = PDFPage.get_pages(
        fp, pagenos, maxpages=maxpages, password=password,
        caching=caching, check_extractable=True)
    for page in pages:
        interpreter.process_page(page)
    fp.close()
    device.close()
    str = retstr.getvalue()
    retstr.close()

    # Formatting removing and replacing special characters
    str = str.replace("\r", "\n")
    # str = re.sub(re.bullet, " ", str)

    return str #str.decode('ascii', errors='ignore')

    return str

def ingest(data_csv, columns,drops):
    df = pd.read_csv(data_csv, encoding="ISO-8859-1")
    df.columns = columns
    df.drop(drops, axis=1, inplace=True, errors='ignore')
    # data.label.replace([0,2,4], [0,1,2], inplace=True)
    df = df[df['message'].isnull() == False]
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    return df

def tokenize_text_column(df, source_column_name, new_column_name, cleanup_function_name):
    if cleanup_function_name == "text":
        df[new_column_name] = df[source_column_name].map(clean_and_tokenize_sentences)
    elif cleanup_function_name == "tweet":
        df[new_column_name] = df[source_column_name].map(clean_and_tokenize_messages)
    df = df[df.tokens != 'NC']
    df.reset_index(inplace=True)
    df.drop('index', inplace=True, axis=1)
    return df

def csv_reader(data_csv, columns, drops, num_messages = 50000):
    df = ingest(data_csv, columns, drops)
    df = df.sample(num_messages)  # Random sample
    return df

def read_document(filepath):
    f = open(filepath)
    raw = f.read()
    f.close()
    return raw

def dir_reader(dir_path):
    docs = os.listdir(dir_path)
    for filename in docs:
        if os.path.isfile(dir_path + filename):
            document = read_document(dir_path + filename)


if __name__ == "__main__":
    # csv = '../data/sentiment/SentimentAnalysisDataset.csv'
    # keep_columns = ['ItemID', 'label', 'SentimentSource', 'message']
    # drop_columns = ['ItemID', 'SentimentSource']
    # df = csv_reader(csv, keep_columns,drop_columns)
    # dfm = tokenize_text_column(df,"message","tokens","tweet")
    # print(dfm.head())
    # dfs = tokenize_text_column(df,"message","tokens","text")
    # print(dfs.head())
    #
    # pdfminer_text = to_text_by_pdfminer("../data/invoices/test1.pdf")
    # print(pdfminer_text)

    tesseract_text = to_text_by_tesseract("../data/invoices/test1.tif")
    print(tesseract_text)