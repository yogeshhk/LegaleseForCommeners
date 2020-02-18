#======================================================================================================================
#   Using LDA for Topic Modeling
#
#   Reference:
#
#   Copyright (C) 2017 Yogesh H Kulkarni
#======================================================================================================================
import warnings
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
from reader import csv_reader, tokenize_text_column
from vectoriser import vectorise_column_bow

import pandas as pd
pd.options.mode.chained_assignment = None

from gensim.models import LdaModel
num_topics_per_doc = 1
num_words_per_topic = 3

saved_corpora = "./corpora.mm"

min_word_length = 2 # allow only bigger words

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

def get_filename_of_doc(map, my_doc):
    for name, doc in map.items():
        if doc == my_doc:
            return name
    return ""

def prep_for_topic_modeling(data_csv, columns, drops, n_samples):
    df = csv_reader(data_csv, columns, drops, n_samples)
    df = tokenize_text_column(df,"message","tokens","text")
    df = df[df.tokens != 'NC']
    df.reset_index(inplace=True)
    df.drop('index', inplace=True, axis=1)

    print("Vectorizing texts...")
    bow_corpus, sentenceId_bow_map, dictionary = vectorise_column_bow(df.tokens)

    return bow_corpus, sentenceId_bow_map, dictionary

if __name__ == "__main__":

    data_csv = '../data/chatmessages/sample_data.csv'
    columns = ['ConversationId', 'PersonId', 'message','Date']
    drops = ['ConversationId', 'PersonId', 'Date']
    n_samples = 200

    bow_corpus, sentenceId_bow_map, dictionary =   prep_for_topic_modeling(data_csv, columns, drops, n_samples)
    ldamodel = LdaModel(corpus=bow_corpus, id2word=dictionary)
    all_model_topics_list = ldamodel.show_topics(formatted=False, num_topics=len(dictionary))
    print("All topics: {}".format(all_model_topics_list))

    # List of all topics, same length as dictionary, indexed first at 0th, second at 1st index in the list...
    # Each doc is made up of all the topics with varying weights, most 0
    # Each topic has distribution of word it represents
    for bow_vec in bow_corpus:
        topics, weights = get_doc_topic_words(ldamodel, all_model_topics_list, bow_vec, num_topics_per_doc, num_words_per_topic)
        filename = get_filename_of_doc(sentenceId_bow_map, bow_vec)
        print("Message: {}, Topics: {} Distribution {}".format(filename, topics, weights))
