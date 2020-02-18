import os
import re
import pandas as pd
from collections import OrderedDict
from operator import itemgetter
import json

def find_articles_in_case(doc):
    current_full_path = current_cases_dir + doc
    content = open(current_full_path, 'r').read()
    matches = re.findall(r'article (\d+)',content.lower())
    matches = matches + re.findall(r'art. (\d+)',content.lower())
    matches = matches + re.findall(r'arts. (\d+)',content.lower())
    return list(set(matches))

def find_prior_cases_having_articles(articles, prior_dict):
    prior_cases = []
    for k,v in prior_dict.items():
        for art in articles:
            search_term = r"art(\.|s\.|icle)? {}".format(art)
            matches = re.findall(search_term, v.lower())
            if len(matches):
                prior_cases.append(k)
                break
    return prior_cases

def find_acts_in_case(doc):
    current_full_path = current_cases_dir + doc
    content = open(current_full_path, 'r').read()
    matches = re.findall(r'\[[a-z,0-9() ]*act[a-z,0-9() ]*\]',content.lower())
    matches = list(set(matches))
    return matches

def find_prior_cases_having_acts(acts, prior_dict):
    prior_cases = []
    for k,v in prior_dict.items():
        for act in acts:
            # search_term = r"art(\.|s\.|icle)? {}".format(act)
            # matches = re.findall(search_term, v.lower())
            if act in v.lower():
                prior_cases.append(k)
                break
    return prior_cases

if __name__ == "__main__":
    regex_citations_dict = OrderedDict()
    # similarity_matrix_file = "./data/similarity.csv"
    #
    # df = None
    # if os.path.isfile(similarity_matrix_file):
    #     print("Loading Dataframe")
    #     df = pd.read_csv(similarity_matrix_file, index_col=0)
    #     # df[df < 0.000000001] = 0.500001 # some correct results have score -0.0018. HARDCODING

    current_cases_dir = "./data/Task_2/Current_Cases/"
    prior_cases_dir = "./data/Task_2/Prior_Cases/"
    current_cases_filenames = [doc for doc in os.listdir(current_cases_dir) if doc.endswith('.txt')]
    prior_cases_filenames = [doc for doc in os.listdir(prior_cases_dir) if doc.endswith('.txt')]

    prior_cases_content_dict = OrderedDict()
    for pri_file in prior_cases_filenames:
        prior_full_path = prior_cases_dir + pri_file
        content = open(prior_full_path, 'r').read()
        prior_cases_content_dict[pri_file] = content

    for cur_file in current_cases_filenames:
        articles = find_articles_in_case(cur_file)
        prior_cases = find_prior_cases_having_articles(articles, prior_cases_content_dict)
        acts = find_acts_in_case(cur_file)
        prior_cases = prior_cases + find_prior_cases_having_acts(acts, prior_cases_content_dict)
        if len(prior_cases) == 0:
            print("Case {} has no citations for keywords {}".format(cur_file,articles))
        regex_citations_dict[cur_file] = list(set(prior_cases))
        # prior_cases_scores = OrderedDict()
        # for pri_file in prior_cases_filenames:
        #     if pri_file in prior_cases:
        #         df.loc[cur_file, pri_file] = df.loc[cur_file,pri_file] + 1
        #     # if score > 0.5:
        #     prior_cases_scores[pri_file] = df.loc[cur_file, pri_file]
        # prior_cases_scores_sorted = sorted(prior_cases_scores.items(), key=itemgetter(1), reverse=True)
        # print("For {} the prior articles are {}".format(cur_file,prior_cases_scores_sorted))

    regex_citations_json = "./data/regex_citations.json"
    with open(regex_citations_json, 'w') as outfile:
        json.dump(regex_citations_dict, outfile, indent=4)

