import pandas as pd
from bs4 import BeautifulSoup
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import nltk
stopwords = set(nltk.corpus.stopwords.words('english'))
from nltk import word_tokenize
from nltk import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

import os
path = '../Datasets/uci_ml_legalcases/test/'

# def compute_similarity(sent1, sent2):
#     sentences = [sent1, sent2]
#
#     from sklearn.feature_extraction.text import CountVectorizer
#     c = CountVectorizer()
#     bow_matrix = c.fit_transform(sentences)  # rows are sentences and the columns are words
#
#     from sklearn.feature_extraction.text import TfidfTransformer
#     normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
#     similarity_graph = normalized_matrix * normalized_matrix.T
#     return (similarity_graph[0,1])

def compute_similarity(sent, catch_words):
    sentence_words = process_sentence(sent)
    common_words = set(sentence_words)&set(catch_words)
    return len(common_words) / (float(len(sentence_words) + len(catch_words)))

def compute_similarity_with_catchphrases(sent, catchphrases):
    return max([compute_similarity(sent,process_sentence(catchphrs)) for catchphrs in catchphrases])


def read_uci_ml_case(filename):
    fullreadfilename = path + filename
    cleaned_sentences = []
    with open(fullreadfilename) as rf:
        for line in rf:
            cleaned_sentences.append(re.sub('\s+', ' ', line.rstrip()).strip())
    page = " ".join(cleaned_sentences)
    soup = BeautifulSoup(page, "html.parser")
    catchphrases = [unicode(ctph.text) for ctph in soup.findAll('catchphrase')]
    sentences = [unicode(sent.text) for sent in soup.findAll('sentence')]
    return catchphrases, sentences

def process_sentence(sentence):
    words = [ wordnet_lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word not in stopwords]
    return words

def prep_cue_words_file(catchphrases, filename):
    wf = open(path + "cuewords/" + filename + ".txt", "w")
    words = []
    for catchph in catchphrases:
        current_words = process_sentence(catchph)
        for word in current_words:
            wf.write(word + "\n")
        words += [word.strip() for word in current_words]
    wf.close()
    return " ".join(set(words))


if __name__ == "__main__":
    master_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith(".xml"): # read only xml files, directories hamper otherwise
            catchphrases, sentences = read_uci_ml_case(filename)
            df = pd.DataFrame.from_dict({'Sentence': sentences})
            words = prep_cue_words_file(catchphrases, filename) # Need 'cuewords' directory, where important words are writtent to a file with same filename
            df['CueWords'] = pd.Series(words, index=df.index)
            df['Similarity'] = [compute_similarity_with_catchphrases(sent, catchphrases) for sent in sentences]
            threshold = (df['Similarity'].max() - df['Similarity'].min()) * 0.95 + df['Similarity'].min()
            df['Important'] = df['Similarity'] > threshold
            master_df = master_df.append(df) # go on appending each df so as to make full master df
            df.to_csv(path + "csvs/" + filename + ".csv") # Need 'csvs' direcotry, where the df is written to filename.csv
    master_df.to_csv(path + "master_df.csv", index=False)
    # pd.set_option('display.max_rows', len(master_df))
    # print(master_df)
    # pd.reset_option('display.max_rows')
