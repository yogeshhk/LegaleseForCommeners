"""
Robot Lawyer Demo App : Finding Judgement Similarity

Author: Yogesh H Kulkarni
Last modified: 18 November 2016
"""
import os
import re
import pandas as pd
import scipy
import nltk
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from gensim.summarization import summarize
import networkx as nx


from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

stopwords = set(nltk.corpus.stopwords.words('english'))

DEBUG = 0
def log(s,level=DEBUG):
    if level:
        print(s)

class robotDataFrame:
    def __init__(self, path_to_judgements_txt_files):
        self.path = path_to_judgements_txt_files #'../Datasets/yhk_cleaned_judgements/test/'

        self.judgements, self.filenames = self.populate_judgements()

        self.find_tfidf_similarity()

        self.df = pd.DataFrame(index=self.filenames, columns=self.filenames)

        self.fill_pairwise_scores()

        self.fill_top5_judgements_filenames()

        self.fill_judgements_content()

        self.update_stopwords()  # once

        self.fill_top_keywords()

        self.fill_kmeans_clusters()

        # identify_lda_topics()

        self.fill_summary("textrank")

        print(self.df.head())

    def read_judgement_from_directory(self,fullreadfilename):

        with open(fullreadfilename) as rf:
            content_in_list = rf.readlines()

        content = " ".join(content_in_list)
        return content

    def populate_judgements(self):
        filenames = []
        judgements = {}
        files = os.listdir(self.path)
        ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
        for id, filename in enumerate(ordered_files):
            content = self.read_judgement_from_directory(self.path + filename)
            filenames.append(filename)
            judgements[filename] = content
            log("Filename {} :\n {}".format(filename,content))
        return judgements, filenames

    def purge_judgement_text(self,judgement):
        raw_words = nltk.word_tokenize(judgement) # or  re.split(' |,|:|\n|\t',content.lower()
        tagged_words = nltk.pos_tag(raw_words)
        noun_words = [i.lower() for (i,category) in tagged_words if "NN" in category]
        words = [i for i in  noun_words if i not in stopwords and len(i) > 3]
        return words

    def find_tfidf_similarity(self):
        raw_judgements = [self.judgements[filename] for filename in self.filenames]
        cleaned_judgements = [self.purge_judgement_text(jd) for jd in raw_judgements]
        documents = [" ".join(cjd) for cjd in cleaned_judgements]
        tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        self.pairwise_scores = self.tfidf_matrix * self.tfidf_matrix.T
        self.feature_names = tfidf_vectorizer.get_feature_names()

    def fill_pairwise_scores(self):
        cx = scipy.sparse.coo_matrix(self.pairwise_scores)
        for i,j,v in zip(cx.row, cx.col, cx.data):
            self.df.iloc[i,j] = v

    def fill_top5_judgements_filenames(self):
        self.df['Top5Similar'] =  pd.Series([[]])
        for index, row in self.df.iterrows():
            top5 = row.sort_values(ascending=False)[1:6].axes
            top5list = [ent for ent in top5[0]]
            self.df.loc[index,'Top5Similar'] = top5list

    def fill_judgements_content(self):
        self.df['Judgement'] = pd.Series([""])
        for index, row in self.df.iterrows():
            self.df.loc[index, "Judgement"] = self.judgements[index]

    def update_stopwords(self):
        path_to_stopwords_txt_file = '../Datasets/legalstopwords.txt'
        custom_keywords = set(line.strip() for line in open(path_to_stopwords_txt_file))
        stopwords.update(custom_keywords)

    def find_keywords(self,judgement):
        words = self.purge_judgement_text(judgement)
        fdist = nltk.probability.FreqDist(words)
        keywords_list = fdist.most_common(10)
        keywords = [word for (word, count) in keywords_list]
        return keywords

    def fill_top_keywords(self):
        self.df['Keywords'] = pd.Series([[]])
        for index, row in self.df.iterrows():
            judgement = self.df.loc[index, "Judgement"]
            keywords = self.find_keywords(judgement)
            self.df.loc[index, "Keywords"] = keywords

    def textrank(self, document):
        sentence_tokenizer = PunktSentenceTokenizer()
        sentences = sentence_tokenizer.tokenize(document)

        bow_matrix = CountVectorizer().fit_transform(sentences)
        normalized = TfidfTransformer().fit_transform(bow_matrix)

        similarity_graph = normalized * normalized.T

        nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
        scores = nx.pagerank(nx_graph)
        ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)),
                      reverse=True)
        return [sent for score, sent in ranked]

    def fill_summary(self,method="gensim"):
        self.df['Summary'] = pd.Series([""])
        for index, row in self.df.iterrows():
            judgement = self.df.loc[index, "Judgement"]
            core_judgement = judgement.split("APPELLATE JURISDICTION")[-1] # main text starts after this split word
            sumlist = []
            if method == "gensim":
                sumlist = summarize(core_judgement, split=True)
            elif method == "textrank":
                sumlist = self.textrank(core_judgement)
            self.df.loc[index, "Summary"] = " ".join(sumlist[:5])

    def fill_kmeans_clusters(self):
        num_clusters = 5
        km = KMeans(n_clusters=num_clusters)
        km.fit(self.tfidf_matrix)
        clusters = km.labels_.tolist()

        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        for i in range(num_clusters):
            centroid = order_centroids[i, :1]
            print("Cluster {} : Centroid : {}".format(i, self.feature_names[centroid[0]]))

            self.df['ClusterID'] = clusters

    def identify_lda_topics(self):
        doc_clean = []
        for index, row in self.df.iterrows():
            judgement = self.df.loc[index, "Judgement"]
            words = self.purge_judgement_text(judgement)
            #doc_clean.append(" ".join(words))
            doc_clean.append(words)

        dictionary = corpora.Dictionary(doc_clean)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        ldamodel = models.ldamodel.LdaModel(doc_term_matrix, num_topics=6, id2word=dictionary, passes=5)
        for topic in ldamodel.show_topics(num_topics=5, formatted=False, num_words=6):
            print("Topic {}: Words: ".format(topic[0]))
            topicwords = [w for (w, val) in topic[1]]
            print(topicwords)

    def querySearchedJudgementFilenamesBySearchItem(self, keyword):
        matching_rows = [index for index, row in self.df.iterrows() if keyword in row['Keywords']]
        return matching_rows

    def querySimilarJudgementFilenamesByGivenJudgement(self, index):
        similar_judmenets = self.df.loc[index,"Top5Similar"]
        return similar_judmenets

    def queryKeywordsByFilename(self,index):
        keywords = self.df.loc[index,"Keywords"]
        return " ".join(keywords[:3])

    def queryFullJudgementByFilename(self,index):
        return self.df.loc[index,"Judgement"]

    def querySummaryByFilename(self,index):
        return self.df.loc[index, "Summary"]

if __name__ == '__main__':
    path = '../Datasets/yhk_cleaned_judgements/test/'
    rDf = robotDataFrame(path)
    # searches = rDf.querySearchedJudgementFilenamesBySearchItem("income")
    # print(searches)
    #
    # similars = rDf.querySimilarJudgementFilenamesByGivenJudgement("supremecourt_4_.txt")
    # print(similars)

    summary = rDf.querySummaryByFilename("supremecourt_4_.txt")
    print(summary)