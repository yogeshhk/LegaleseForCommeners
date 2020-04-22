import pandas as pd
import pickle as pk
import os
import json
from collections import OrderedDict
import nltk
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sklearn.feature_extraction.text as text
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Save stopwords into set for faster lookup
stops = set(stopwords.words('english'))
path_to_stopwords_txt_file = './data/legalstopwords.txt'
custom_keywords = set(line.strip() for line in open(path_to_stopwords_txt_file))
stops.update(custom_keywords)

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


import warnings
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

import pandas as pd
import numpy as np
from nltk.corpus import words

pd.options.mode.chained_assignment = None
from gensim.corpora import Dictionary
from gensim.models import LdaModel
num_topics_per_doc = 5
num_words_per_topic = 3
id_column = "FileId"
content_column = "Content"
tokens_column = "Tokens"
min_word_length = 2 # allow only bigger words

def vectorise_column_tfidf(pandas_fileId_column, pandas_wordlist_column):
    text_list = np.array(pandas_wordlist_column)
    file_Id_list = np.array(pandas_fileId_column)

    dictionary = Dictionary(text_list)
    additional_documents = []
    additional_documents.append(words.words())
    dictionary.add_documents(additional_documents)
    dictionary.compactify()

    list_of_review_texts = [" ".join(tokens) for tokens in text_list]
    vectorizer = TfidfVectorizer(min_df=5).fit(list_of_review_texts)
    print(len(vectorizer.get_feature_names()))

    text_list_vectorized = vectorizer.transform(list_of_review_texts)
    feature_names = np.array(vectorizer.get_feature_names())
    sorted_tfidf_index = text_list_vectorized.max(0).toarray()[0].argsort()
    n = int(len(feature_names)/10)
    largest_tfidf_coeff_words = feature_names[sorted_tfidf_index[:-1*n:-1]]
    print(largest_tfidf_coeff_words)
    sentenceId_bow_map = {}
    bow_corpus = []
    for i, tokens in zip(file_Id_list,text_list):
        intersection_words = list(set(tokens) & set(largest_tfidf_coeff_words))
        bow = dictionary.doc2bow(intersection_words)
        # if len(intersection_words):
        #     print("PaperId: {}, Imp words: {}".format(i,intersection_words))
        #     freq_words_string = " ".join(intersection_words)
        #     response = vectorizer.transform([freq_words_string])
        #     bow = [response[0, col] for col in response.nonzero()[1]]
        #     print(bow)
        # else:
        #     bow = [0.0]
        bow_corpus.append(bow)
        sentenceId_bow_map[i] = bow

    return bow_corpus, sentenceId_bow_map, dictionary

def vectorise_column_bow(pandas_fileId_column, pandas_tokens_column):
    text_list = np.array(pandas_tokens_column)
    fileId_list = np.array(pandas_fileId_column)
    # Makes a list all unique words in all the lists supplied and indexes it like an array, id2word dictionary
    dictionary = Dictionary(text_list)
    additional_documents = []
    additional_documents.append(words.words())
    dictionary.add_documents(additional_documents)
    dictionary.compactify()
    # dictionary.save(saved_dictionary)  # store the dictionary, for future reference
    # print("Dictionary saved at {}".format(saved_dictionary))

    sentenceId_bow_map = {}
    bow_corpus = []
    for i, text in zip(fileId_list,text_list):
        bow = dictionary.doc2bow(text)
        bow_corpus.append(bow)
        sentenceId_bow_map[i] = bow

    return bow_corpus, sentenceId_bow_map, dictionary

# Ref: https://groups.google.com/forum/#!topic/gensim/CJZc7KN60JE
def get_doc_topic_words(model, all_model_topics_list, doc, num_topics=1,num_words_per_topic=2):
    topic_words = {}
    doc_wise_topic_distribution = model.__getitem__(doc, eps=0.0001) #  like [(1, 0.8), (50,0.2)],
    # which means topic #1 has weight 0.8 in the document, while topic #50 0.2. All other topics have implicit 0.0.
    # By eps we are filtering those with value less than eps,
    # this list of tuples is not sorted, so the first one is not necessarily hving highest weight, so SORT
    sorted_doc_wise_topics = sorted(doc_wise_topic_distribution, key=lambda x: x[1],reverse=True)
    doc_wise_weighted_values = []#[0] * num_words_per_topic # initial array of 0s of num_words_per_topic dimension
    for topic_id, weight in sorted_doc_wise_topics[0:num_topics]:
        topic_word_distribution = model.state.get_lambda()[topic_id]
        # topic_i_id, topic_wordlist = all_model_topics_list[topic_id] #  Gives on 10 words per topc, not sure if they are top most
        topic_word_distribution = topic_word_distribution / topic_word_distribution.sum()  # normalize to probability dist
        topic_wordlist = [(index, value) for index, value in enumerate(topic_word_distribution)]
        sorted_topic_wordlist = sorted(topic_wordlist, key=lambda x: x[1],reverse=True)
        topic_wise_word_weights = [prob*weight for position, prob in sorted_topic_wordlist[0:num_words_per_topic]]
        #topic_wise_word_values = [position for position, prob in sorted_topic_wordlist[0:num_words_per_topic]]
        #doc_wise_weighted_values = [x + y for x, y in zip(doc_wise_weighted_values, topic_wise_word_weights)]
        doc_wise_weighted_values = doc_wise_weighted_values + topic_wise_word_weights
        short_topic_wordlist = [dictionary[int(position)] for position, prob in sorted_topic_wordlist[0:num_words_per_topic]]
        topic_words[topic_id] = short_topic_wordlist
    return topic_words, doc_wise_weighted_values

def get_topic_to_wordids(model):
    p = list()
    for topicid in range(model.num_topics):
        topic = model.state.get_lambda()[topicid]
        topic = topic / topic.sum() # normalize to probability dist
        p.append(topic)
    return p

def get_fileId_of_content(map, my_doc):
    for name, doc in map.items():
        if doc == my_doc:
            return name
    return ""

def prep_for_topic_modeling(df):
    df[tokens_column] = df[content_column].map(conent_to_wordlist)
    df = df[df.Tokens != 'NC']
    df.reset_index(inplace=True)
    df.drop('index', inplace=True, axis=1)

    print("Vectorizing texts...")
    bow_corpus, sentenceId_bow_map, dictionary = vectorise_column_bow(df[id_column],df[tokens_column])
    # bow_corpus, sentenceId_bow_map, dictionary = vectorise_column_tfidf(df[id_column],df[tokens_column])

    return bow_corpus, sentenceId_bow_map, dictionary

def conent_to_wordlist(doc, remove_stopwords=True):
    try: ## YHK
        # Function converts text to a sequence of words,
        # Returns a list of words.

        lemmatizer = WordNetLemmatizer()
        # 1. Remove non-letters
        review_text = re.sub("[^a-zA-Z]", " ", doc)
        #     review_text = unicode(review_text, errors='ignore')
        #     review_text = review_text.decode('cp1252').encode('utf-8', errors='ignore')

        # review_text = review_text.decode('utf-8', errors='ignore')

        # 2. Convert words to lower case and split them
        words = review_text.lower().split()
        # 3. Remove stop words
        words = [w for w in words if not w in stops]
        # 4. Remove short words
        words = [t for t in words if len(t) > 2]
        # 5. lemmatizing
        words = [nltk.stem.WordNetLemmatizer().lemmatize(t) for t in words]

        return (words)
    except: ## YHK
        return 'NC'


if __name__ == "__main__":
    current_cases_dir = "./data/Task_2/Current_Cases/"
    prior_cases_dir = "./data/Task_2/Prior_Cases/"
    lda_citations_json = "./data/lda_citations.json"
    lda_citations_dict = None

    if not os.path.isfile(lda_citations_json):
        current_cases_filenames = [doc for doc in os.listdir(current_cases_dir) if doc.endswith('.txt')]
        prior_cases_filenames = [doc for doc in os.listdir(prior_cases_dir) if doc.endswith('.txt')]

        all_docs_dict = OrderedDict()
        current_cases_content_dict = OrderedDict()
        for cur_file in current_cases_filenames:
            print("Processing {} ...".format(cur_file))
            cur_full_path = current_cases_dir + cur_file
            content = open(cur_full_path, 'r').read()
            current_cases_content_dict[cur_file] = content
            all_docs_dict[cur_file] = content

        prior_cases_content_dict = OrderedDict()
        for pri_file in prior_cases_filenames:
            print("Processing {} ...".format(pri_file))
            prior_full_path = prior_cases_dir + pri_file
            content = open(prior_full_path, 'r').read()
            prior_cases_content_dict[pri_file] = content
            all_docs_dict[pri_file] = content

        df = pd.DataFrame(list(all_docs_dict.items()), columns=[id_column, content_column])
        # print(df.head())

        bow_corpus, fileId_bow_map, dictionary = prep_for_topic_modeling(df)
        ldamodel = LdaModel(corpus=bow_corpus, id2word=dictionary)
        all_model_topics_list = ldamodel.show_topics(formatted=False, num_topics=len(dictionary))
        print("All topics: {}".format(all_model_topics_list))
        # List of all topics, same length as dictionary, indexed first at 0th, second at 1st index in the list...
        # Each doc is made up of all the topics with varying weights, most 0
        # Each topic has distribution of word it represents
        lda_citations_dict = OrderedDict()
        for bow_vec in bow_corpus:
            topics, weights = get_doc_topic_words(ldamodel, all_model_topics_list, bow_vec, num_topics_per_doc,
                                                  num_words_per_topic)
            fileId = get_fileId_of_content(fileId_bow_map, bow_vec)
            print("{}: {}, Topics: {} ".format(id_column, fileId, topics))
            lda_citations_dict[fileId] = [v for l in topics.values() for v in l ]

        print("Saving LDA citations...")
        with open(lda_citations_json, 'w') as outfile:
            json.dump(lda_citations_dict, outfile)
    else:
        print("Opening LDA citations...")
        with open(lda_citations_json, 'r') as infile:
            lda_citations_dict = json.load(infile)
