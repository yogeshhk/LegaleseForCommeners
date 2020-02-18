#======================================================================================================================
#   Generates word vectors using gensim
#   Copyright (C) 2017 Yogesh H Kulkarni
#======================================================================================================================
import gensim
import os
import json

from generate_inputdataframe import generate_dataframe

from gensim.models.doc2vec import TaggedDocument

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

input_folder_path = "D:/Education/DataScience/DataSets/Text/Legal/nd_judgements/"
doc2vec_model = "./data/doc2vec.model"
id_column = 'doc_id'
content_column = 'doc_content'
vector_column = 'word2vec'
tag2doc_json = "./data/tag2doc.json"


def clean_doc(content):
    return [wrd for wrd in tokenizer.tokenize(content) if len(wrd) >= 3 and wrd not in stops]

def generate_tagged_docs(dict_df):
    tags = dict_df[id_column]
    contents = dict_df[content_column]
    tag2doc = {}
    taggeddocs = []
    for name, content in zip(tags,contents):
        tag = u'{}'.format(name)
        wordlist = clean_doc(content)
        tdoc = TaggedDocument(words=wordlist, tags=[tag])
        tag2doc[tag] = wordlist
        taggeddocs.append(tdoc)
    return tag2doc, taggeddocs

def generate_doc2vecs(folder_path):
    dict_df = generate_dataframe(folder_path)
    if not os.path.isfile(doc2vec_model):
        tag2doc, taggeddocs = generate_tagged_docs(dict_df)
        print("Building model...")
        model = gensim.models.Doc2Vec(size=100, alpha=0.025, min_alpha=0.025, min_count=1, dm=0)
        model.build_vocab(taggeddocs)
        for epoch in range(20):
            if epoch % 5 == 0:
                print('Now training epoch %s' % epoch)
            model.train(taggeddocs, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(doc2vec_model)
        with open(tag2doc_json, 'w') as outfile:
            json.dump(tag2doc, outfile)
    else:
        print("Retreiving model...")
        model = gensim.models.Doc2Vec.load(doc2vec_model)
        with open(tag2doc_json, 'r') as infile:
            tag2doc = json.load(infile)

    print("Doc2Vec model ready")

    vectors = []
    for cur_doc in dict_df[id_column]:
        wordlist1 = tag2doc[cur_doc]
        vec1 = model.infer_vector(wordlist1, alpha=0.025, min_alpha=0.025, steps=20)
        vectors.append(vec1)

    dict_df[vector_column] = vectors
    print(dict_df.head())
    return model, tag2doc, dict_df

if __name__ == "__main__":
    model, tag2doc, dict_df = generate_doc2vecs(input_folder_path)
