import subprocess
import json
import os
from collections import OrderedDict
import pandas as pd
import operator

def extract_keywords(core_nlp_path, ner_jar_path, filename):
    print("Testing and Extracting from {} ...".format(filename))
    command = core_nlp_path + ner_jar_path + " " + filename
    commandlist = command.split()
    result = subprocess.run(commandlist,stdout=subprocess.PIPE)
    result = result.stdout.decode('cp1252')
    # print(result)
    keywords = {}
    current_word = ""
    current_score = []
    for line in result.split("\n"):
        words = line.split()
        if len(words) != 4:
            continue
        word, _,_, tag = line.split()
        tag,score = tag.split("/")
        if tag == "B-LEGAL":
            current_word = word
            current_score.append(float(score))
        elif tag == "I-LEGAL":
            current_word = current_word + " " + word
            current_score.append(float(score))
        elif current_word != "":
            keywords[current_word] = sum(current_score)/len(current_score)
            # print("Keyword: {}".format(current_word))
            current_word = ""
            current_score = []
    return sorted(keywords.items(), key=operator.itemgetter(1), reverse=True)

def add_dummy_O(test_conll_dir, test_doc):
    train_statement_filename = test_conll_dir + test_doc
    train_statement_O_filename = test_conll_dir + "O/O_" + test_doc
    train_file_df = pd.read_csv(train_statement_filename, encoding='cp1252', header=None)
    train_file_df[len(train_file_df.columns)] = 'O'
    train_file_df.to_csv(train_statement_O_filename, encoding='cp1252', index=None, sep=" ", header=None)
    return train_statement_O_filename


if __name__ == "__main__":
    submission_id = "rightstepspune"
    test_conll_dir = "./data/Task_1/Test_conll/"
    core_nlp_jar_path = "D:/Education/DataScience/NLP/Mining/codeByOthers/CRFpp/CRF++-0.58/crf_test  -v1 -m "
    trained_ner_path = "D:/Education/DataScience/NLP/Mining/codeByOthers/CRFpp/CRF++-0.58/example/fire2017/model "
    track1_results_json = "./data/track1_crfpp_results.json"
    track1_results_submission = "./data/track1_submission_results.txt"


    test_docs = os.listdir(test_conll_dir)
    results_dict = OrderedDict()
    for test_doc in test_docs:
        print("Processing {} ...".format(test_doc))
        if not test_doc.endswith(".txt"):
            continue
        test_statement_filename = test_conll_dir + test_doc
        test_statement_dummy_filename = add_dummy_O(test_conll_dir, test_doc)
        result = extract_keywords(core_nlp_jar_path,trained_ner_path,test_statement_dummy_filename)
        print(result)
        results_dict[test_doc] = result

    print(results_dict)
    print("Saving Track 1 results..")
    with open(track1_results_json, 'w') as joutfile:
        json.dump(results_dict, joutfile, indent=4)

    print("Saving Track 1 submission results..")
    with open(track1_results_submission, 'w') as toutfile:
        for k,v in results_dict.items():
            filename = k.replace("conll.txt","statement")
            catchphrases = ["{}:{}".format(kk,vv) for (kk,vv) in v]
            line = submission_id + " || " + filename +  " || " + ",".join(catchphrases) + "\n"
            toutfile.write(line)
