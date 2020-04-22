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

import gensim
from gensim.models.doc2vec import TaggedDocument
def build_model(judgements):
    taggeddocs = []
    for key, value in judgements.items():
        taggeddocs.append(TaggedDocument(words=value, tags=[key]))

    # build the model
    model = gensim.models.Doc2Vec(taggeddocs, dm=0, size=20, min_count=0, iter=50)

    # training
    for epoch in range(80):
        if epoch % 20 == 0:
            log('Now training epoch %s' % epoch, 1)
        model.train(taggeddocs)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    return model

judgements, df = populate_judgements(path)
model = build_model(judgements)

import numpy
dimension = len(judgements)
SimilarityScores = numpy.zeros((dimension,dimension))
for id1, key1 in enumerate(df.index):
    log('Row id {} file {}'.format(id1, key1))
    text1 = judgements[key1]
    for id2, key2 in enumerate(df.columns):
        log('Column id {} file {}'.format(id2,key2))
        text2 = judgements[key2]
        if id2 > id1:
            score =   model.n_similarity(text1,text2)
            g = float("{0:.2f}".format(score))
            log("{} vs {} = {}".format(id1,id2,g))
            SimilarityScores[id1,id2] = g

max_indices_in_rows = SimilarityScores.argmax(axis=0) # Find row wise max score
log(max_indices_in_rows)
for id, file in enumerate(df.index):
    max = max_indices_in_rows[id]
    log("File {} has most similarity with {}".format(file,df.columns[max]),1)

