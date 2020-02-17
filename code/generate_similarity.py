# Generates similarity matrix amongst the docs
# Input: Dictionary with doc_id as key and doc_content as values as a string

from generate_word2vecs import generate_doc2vecs
import pandas as pd
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

input_folder_path = "D:/Education/DataScience/DataSets/Text/Legal/nd_judgements/"
doc2vec_model = "./data/doc2vec.model"
id_column = 'doc_id'
content_column = 'doc_content'
vector_column = 'word2vec'
tag2doc_json = "./data/tag2doc.json"

def generate_similarity(folder_path):
    model, tag2doc, dict_df = generate_doc2vecs(input_folder_path)
    vectors = dict_df[vector_column]
    filenames = dict_df[id_column]

    print("Calculating similarities")
    similarity_matrix = OrderedDict()
    for cur_doc, vec1 in zip(filenames, vectors):
        similarities = []
        for pri_doc, vec2 in zip(filenames, vectors):
            similarity = cosine_similarity([vec1], [vec2])[0]
            similarities.append(similarity[0])
        similarity_matrix[cur_doc] = similarities

    print("Making Dataframe")
    df = pd.DataFrame.from_dict(similarity_matrix)
    df.columns = filenames
    print(df.head())
    dict_df = pd.concat([dict_df, df], axis=1)
    return dict_df

if __name__ == "__main__":
    df = generate_similarity(input_folder_path)
    print(df.head())