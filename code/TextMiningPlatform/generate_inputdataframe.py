#======================================================================================================================
#   Reads many types of sources into pandas dataframe with doc_id/filenames as rows and doc_content as columns
#   References:
#       https://raw.githubusercontent.com/skcript/cvscan/master/cvscan/converter.py
#   Copyright (C) 2017 Yogesh H Kulkarni
#======================================================================================================================
import os
import pandas as pd

input_folder_path = "D:/Education/DataScience/DataSets/Text/Legal/nd_judgements/"
id_column = 'doc_id'
content_column = 'doc_content'

def read_document(filepath):
    f = open(filepath)
    raw = f.read()
    f.close()
    return raw

def generate_dictionary(dir_path):
    doc_id_content_dict = {}
    docs = os.listdir(dir_path)
    for filename in docs:
        if os.path.isfile(dir_path + filename):
            raw_text = read_document(dir_path + filename)
            doc_id_content_dict[filename] = raw_text
    return doc_id_content_dict

def generate_dataframe(dir_path):
    data_dict = generate_dictionary(dir_path)
    dict_df = pd.DataFrame()
    dict_df[id_column] = data_dict.keys()
    dict_df[content_column] = data_dict.values()
    # df = pd.DataFrame(data_dict.items(), columns=["doc_id", "doc_content"])
    print(dict_df.head())
    return dict_df



if __name__ == "__main__":
    df = generate_dataframe(input_folder_path)

