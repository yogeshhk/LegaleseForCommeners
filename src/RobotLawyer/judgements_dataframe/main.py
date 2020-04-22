import os
import re
import pandas as pd
import numpy as np
import scipy
import nltk
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from gensim.summarization import summarize

stopwords = set(nltk.corpus.stopwords.words('english'))

DEBUG = 0
#add custom words
def log(s,level=DEBUG):
    if level:
        print(s)

def read_judgement_from_directory(fullreadfilename):
    with open(fullreadfilename) as rf:
        content_in_list = rf.readlines()

    content = " ".join(content_in_list)
    return content

def populate_judgements(path):
    filenames = []
    judgements = {}
    files = os.listdir(path)
    ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
    for id, filename in enumerate(ordered_files):
        content = read_judgement_from_directory(path + filename)
        filenames.append(filename)
        judgements[filename] = content
        log("Filename {} :\n {}".format(filename,content))
    return judgements, filenames

def purge_judgement_text(judgement):
    raw_words = nltk.word_tokenize(judgement) # or  re.split(' |,|:|\n|\t',content.lower()
    tagged_words = nltk.pos_tag(raw_words)
    noun_words = [i.lower() for (i,category) in tagged_words if "NN" in category]
    words = [i for i in  noun_words if i not in stopwords and len(i) > 3]
    return words

def find_tfidf_similarity(judgements, filenames):
    raw_judgements = [judgements[filename] for filename in filenames]
    cleaned_judgements = [purge_judgement_text(jd) for jd in raw_judgements]
    documents = [" ".join(cjd) for cjd in cleaned_judgements]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    pairwise_scores = tfidf_matrix * tfidf_matrix.T
    feature_names = tfidf_vectorizer.get_feature_names()
    return pairwise_scores, tfidf_matrix, feature_names

def fill_pairwise_scores(df,pairwise_scores):
    cx = scipy.sparse.coo_matrix(pairwise_scores)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        df.iloc[i,j] = v
    return df

def fill_top5_judgements_filenames(df):
    df['Top5Similar'] =  pd.Series([[]])
    for index, row in df.iterrows():
        top5 = row.sort_values(ascending=False)[1:6].axes
        df.loc[index,'Top5Similar'] = top5
    return df

def fill_judgements_content(df, judgements):
    df['Judgement'] = pd.Series([""])
    for index, row in df.iterrows():
        df.loc[index, "Judgement"] = judgements[index]
    return df

def update_stopwords():
    path_to_stopwords_txt_file = '../Datasets/legalstopwords.txt'
    custom_keywords = set(line.strip() for line in open(path_to_stopwords_txt_file))
    stopwords.update(custom_keywords)

def find_keywords(judgement):
    words = purge_judgement_text(judgement)
    fdist = nltk.probability.FreqDist(words)
    keywords_list = fdist.most_common(10)
    keywords = [word for (word, count) in keywords_list]
    return keywords

def fill_top_keywords(df):
    df['Keywords'] = pd.Series([[]])
    for index, row in df.iterrows():
        judgement = df.loc[index, "Judgement"]
        keywords = find_keywords(judgement)
        df.loc[index, "Keywords"] = keywords
    return df

def fill_summary(df):
    df['Summary'] = pd.Series([""])
    for index, row in df.iterrows():
        judgement = df.loc[index, "Judgement"]
        core_judgement = judgement.split("APPELLATE JURISDICTION")[-1] # main text starts after this split word
        sum = summarize(core_judgement)
        df.loc[index, "Summary"] = sum
    return df

def fill_kmeans_clusters(df,tfidf_matrix, feature_names):
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        centroid = order_centroids[i, :1]
        print("Cluster {} : Centroid : {}".format(i, feature_names[centroid[0]]))

    df['ClusterID'] = clusters
    return df

def identify_lda_topics(df):
    doc_clean = []
    for index, row in df.iterrows():
        judgement = df.loc[index, "Judgement"]
        words = purge_judgement_text(judgement)
        #doc_clean.append(" ".join(words))
        doc_clean.append(words)

    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    ldamodel = models.ldamodel.LdaModel(doc_term_matrix, num_topics=6, id2word=dictionary, passes=5)
    for topic in ldamodel.show_topics(num_topics=5, formatted=False, num_words=6):
        print("Topic {}: Words: ".format(topic[0]))
        topicwords = [w for (w, val) in topic[1]]
        print(topicwords)
    return df

if __name__ == '__main__':

    path_to_judgements_txt_files = '../Datasets/yhk_cleaned_judgements/test/'

    judgements, filenames = populate_judgements(path_to_judgements_txt_files)

    pairwise_scores, tfidf_matrix, feature_names = find_tfidf_similarity(judgements, filenames)

    df = pd.DataFrame(index=filenames, columns=filenames)

    df = fill_pairwise_scores(df,pairwise_scores) # Fill df (which is filename x filename matrix as of now) with similarity scores

    df = fill_top5_judgements_filenames(df) # Per row, find top 5 similarity scores, find their indices ie filenames, store as list

    df = fill_judgements_content(df, judgements) # Store actual judgment content in a separate column

    update_stopwords() # once

    df = fill_top_keywords(df)

    df = fill_kmeans_clusters(df, tfidf_matrix, feature_names)

    # identify_lda_topics(df)
    df = fill_summary(df)

    print(df.head())


