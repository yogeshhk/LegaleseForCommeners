import os
path = '../Datasets/yhk_cleaned_judgements/test/'
DEBUG = 0
def log(s,level=DEBUG):
    if level:
        print(s)

def read_judgement_from_directory(filename):
    fullreadfilename = path + filename
    with open(fullreadfilename) as rf:
        content_in_list = rf.readlines()

    content = " ".join(content_in_list)
    return content

import re
import pandas as pd
def populate_judgements(path):
    filenames = []
    judgements = {}
    files = os.listdir(path)
    ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
    for id, filename in enumerate(ordered_files):
        content = read_judgement_from_directory(filename)
        filenames.append(filename)
        judgements[filename] = content
        log("Filename {} :\n {}".format(filename,content))
    df = pd.DataFrame(index=filenames, columns=filenames)
    return judgements, df

from sklearn.feature_extraction.text import TfidfVectorizer
def build_model(judgements,df):
    documents = [judgements[filename] for filename in df.index]
    # no need to normalize, since Vectorizer will return normalized tf-idf
    tfidf = TfidfVectorizer().fit_transform(documents)
    pairwise_scores = tfidf * tfidf.T
    return pairwise_scores

judgements, df = populate_judgements(path)
pairwise_scores = build_model(judgements,df)
import numpy, scipy
dimension = len(judgements)
SimilarityScores = numpy.zeros((dimension,dimension))
cx = scipy.sparse.coo_matrix(pairwise_scores)
for i,j,v in zip(cx.row, cx.col, cx.data):
    if j > i:
        SimilarityScores[i,j] = v
log(SimilarityScores,1)
max_indices_in_rows = SimilarityScores.argmax(axis=1) # Find row wise max score
for id, file in enumerate(df.index):
    max = max_indices_in_rows[id]
    log(max)
    log("File {} has most similarity with {}".format(file,df.columns[max]),1)

