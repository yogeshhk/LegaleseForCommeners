"""
    Generating Doc2Vec vectors for current and prior cases
    Steps:
        Get every case as cleaned text, split it to form list of words/tokens, for both, current and prior cases
        Created gensim TaggedDocument for each, giving filename as tag
        Kept map of tag to the content ie wordlist in each case. It is pickled so as not to generate it again
        LDA model is built and saved
"""

import os
import pandas as pd
import gensim
import pickle
import numpy as np
from collections import OrderedDict
from gensim.models.doc2vec import TaggedDocument
import json
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

def clean_doc(content):
    return [wrd for wrd in tokenizer.tokenize(content) if len(wrd) >= 3 and wrd not in stops]


if __name__ == "__main__":
    current_cases_dir = "./data/Task_2/Current_Cases/"
    prior_cases_dir = "./data/Task_2/Prior_Cases/"
    doc2vec_model = "./data/doc2vec.model"
    tag2doc_json = "./data/tag2doc.json"
    similarity_matrix_file = "./data/similarity.csv"


    current_cases_filenames = [doc for doc in os.listdir(current_cases_dir) if doc.endswith('.txt')]
    prior_cases_filenames = [doc for doc in os.listdir(prior_cases_dir) if doc.endswith('.txt')]

    model = None
    tag2doc = {}

    if not os.path.isfile(doc2vec_model):
        taggeddocs = []
        tags = []
        for doc in current_cases_filenames:
            current_full_path = current_cases_dir + doc
            content = open(current_full_path, 'r').read()
            tag = u'{}'.format(doc)
            print("Processing {} ...".format(tag))
            wordlist = clean_doc(content)
            tdoc = TaggedDocument(words=wordlist, tags=[tag])
            tag2doc[tag] = wordlist
            taggeddocs.append(tdoc)

        for doc in prior_cases_filenames:
            prior_full_path = prior_cases_dir + doc
            content = open(prior_full_path, 'r').read()
            tag = u'{}'.format(doc)
            print("Processing {} ...".format(doc))
            wordlist = clean_doc(content)
            tdoc = TaggedDocument(words=wordlist, tags=[tag])
            tag2doc[tag] = wordlist
            taggeddocs.append(tdoc)

        print("Building model...")
        model = gensim.models.Doc2Vec(size=100, alpha=0.025, min_alpha=0.025, min_count=1, dm=0)
        model.build_vocab(taggeddocs)
        for epoch in range(20):
            if epoch % 5 == 0:
                print('Now training epoch %s' % epoch)
            model.train(taggeddocs, total_examples=model.corpus_count)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        model.save(doc2vec_model)
        with open(tag2doc_json, 'w') as outfile:
            json.dump(tag2doc, outfile)
    else:
        model = gensim.models.Doc2Vec.load(doc2vec_model)
        with open(tag2doc_json, 'r') as infile:
            tag2doc = json.load(infile)

    print("Doc2Vec model ready")


    df = None
    if not os.path.isfile(similarity_matrix_file):
        current_case_vectors_file = "./data/current.vecs"
        prior_case_vectors_file = "./data/prior.vecs"

        print("Infer vectors for each case and pickle them")
        current_case_vectors = []
        if not os.path.isfile(current_case_vectors_file):
            with open(current_case_vectors_file, "wb") as fc:
                print("Storing Case Vectors")
                for cur_doc in current_cases_filenames:
                    wordlist1 = tag2doc[cur_doc]
                    vec1 = model.infer_vector(wordlist1, alpha=0.025, min_alpha=0.025, steps=20)
                    current_case_vectors.append(vec1)
                pickle.dump(current_case_vectors, fc)
        else:
            with open(current_case_vectors_file, "rb") as fc:
                current_case_vectors = pickle.load(fc)

        prior_case_vectors = []
        if not os.path.isfile(prior_case_vectors_file):
            with open(prior_case_vectors_file, "wb") as fp:
                print("Storing Prior Vectors")
                for pri_doc in prior_cases_filenames:
                    wordlist2 = tag2doc[pri_doc]
                    vec2 = model.infer_vector(wordlist2, alpha=0.025, min_alpha=0.025, steps=20)
                    prior_case_vectors.append(vec2)
                pickle.dump(prior_case_vectors, fp)
        else:
            with open(prior_case_vectors_file, "rb") as fp:
                prior_case_vectors = pickle.load(fp)


        print("Calculating similarities")
        similarity_matrix = OrderedDict()
        for cur_doc, vec1 in zip(current_cases_filenames, current_case_vectors):
            prior_cases_similarities = []
            for pri_doc,vec2 in zip(prior_cases_filenames,prior_case_vectors):
                similarity = cosine_similarity([vec1], [vec2])[0]
                prior_cases_similarities.append(similarity[0])
            similarity_matrix[cur_doc] = prior_cases_similarities

        print("Making Dataframe")
        df = pd.DataFrame.from_dict(similarity_matrix, orient='index')
        df.columns = prior_cases_filenames
        df.to_csv(similarity_matrix_file)
    else:
        print("Loading Dataframe")
        df = pd.read_csv(similarity_matrix_file, index_col=0)

    print("Results...")
    nlargest = 4
    order = np.argsort(-df.values, axis=1)[:, :nlargest]
    result = pd.DataFrame(df.columns[order],
                          columns=['top{}'.format(i) for i in range(1, nlargest + 1)],
                          index=df.index)

    print(result)
    # similarity_citations_dict = OrderedDict()
    # for index, row in df.iterrows():
    #     print("Index : {}, Row:{}".format(index,row))
    # # similarity_citations_json = "./data/similarity_citations.json"
    # # with open(similarity_citations_json, 'w') as outfile:
    # #     json.dump(similarity_citations_dict, outfile)