import os
import json
import pandas as pd
from collections import OrderedDict
from operator import itemgetter


if __name__ == "__main__":

    results = OrderedDict()
    track2_results_file  = "./data/track2_results.json"

    regex_citations_dict = None
    regex_citations_json = "./data/regex_citations.json"
    with open(regex_citations_json, 'r') as infile:
        regex_citations_dict = json.load(infile)

    lda_citations_dict = None
    lda_citations_json = "./data/lda_citations.json"
    with open(lda_citations_json, 'r') as infile:
        lda_citations_dict = json.load(infile)

    similarity_matrix_file = "./data/similarity.csv"
    df = None
    if os.path.isfile(similarity_matrix_file):
        print("Loading Dataframe")
        df = pd.read_csv(similarity_matrix_file, index_col=0)

    current_cases_dir = "./data/Task_2/Current_Cases/"
    prior_cases_dir = "./data/Task_2/Prior_Cases/"
    current_cases_filenames = [doc for doc in os.listdir(current_cases_dir) if doc.endswith('.txt')]
    prior_cases_filenames = [doc for doc in os.listdir(prior_cases_dir) if doc.endswith('.txt')]
    for cur_file in current_cases_filenames:

        # Regex based prior cases
        prior_cases = regex_citations_dict[cur_file]
        for pri_file in prior_cases_filenames:
            if pri_file in prior_cases:
                current_score = df.loc[cur_file,pri_file]
                df.loc[cur_file, pri_file] = current_score + 1

        # LDA based score
        current_file_topics_set = set(lda_citations_dict.get(cur_file,[]))

        if len(current_file_topics_set) == 0:
            continue

        for pri_file in prior_cases_filenames:
            prior_file_topics_set = set(lda_citations_dict.get(pri_file,[]))
            common_words = list(current_file_topics_set.intersection(prior_file_topics_set))
            total_words = len(current_file_topics_set) + len(prior_file_topics_set)
            score = len(common_words)/total_words
            current_score = df.loc[cur_file, pri_file]
            df.loc[cur_file, pri_file] = current_score + score

        prior_cases_scores = OrderedDict()
        for pri_file in prior_cases_filenames:
            prior_cases_scores[pri_file] = df.loc[cur_file, pri_file]

        prior_cases_scores_sorted = sorted(prior_cases_scores.items(), key=itemgetter(1), reverse=True)
        print("For {} the prior articles are {}".format(cur_file,prior_cases_scores_sorted))
        results[cur_file] = prior_cases_scores_sorted

    print("Saving results ...")
    with open(track2_results_file, 'w') as joutfile:
        json.dump(results, joutfile, indent=4)

    submission_id = "rightstepspune"
    track2_results_submission = "./data/track2_submission_results.txt"
    print("Saving Track 2 submission results..")
    with open(track2_results_submission, 'w') as toutfile:
        for k,v in results.items():
            filename = k.replace(".txt","")
            for id, lst in enumerate(v):
                prfilename = lst[0]
                prfilename = prfilename.replace(".txt","")
                prfilescore = lst[1]
                line = filename +  " Q0 " + "{} {} {} ".format(prfilename,id,prfilescore) + submission_id + "\n"
                toutfile.write(line)