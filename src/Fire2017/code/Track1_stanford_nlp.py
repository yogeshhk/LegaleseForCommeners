import subprocess
import json
import os
from collections import OrderedDict

def extract_keywords(core_nlp_path, ner_jar_path, filename):
    command = "java -mx6g -cp {} edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier {} -textFile {} -outputFormat tsv".format(core_nlp_path,ner_jar_path,filename)
    commandlist = command.split()
    result = subprocess.run(commandlist,stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    # print(result)
    keywords = []
    current_word = ""
    for line in result.split("\n"):
        words = line.split()
        if len(words) != 2:
            continue
        word, tag = line.split()
        if tag == "B-LEGAL":
            current_word = word
        elif tag == "I-LEGAL":
            current_word = current_word + " " + word
        elif current_word != "":
            keywords.append(current_word)
            # print("Keyword: {}".format(current_word))
            current_word = ""

    return list(set(keywords))

if __name__ == "__main__":

    test_conll_dir = "./data/Task_1/Test_conll/"
    core_nlp_jar_path = "D:/Education/DataScience/NLP/Mining/codeByOthers/StanfordNLP/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar"
    trained_ner_path = "D:/Education/DataScience/NLP/Mining/codeByOthers/StanfordNLP/ner-model.ser.gz"
    track1_results_json = "./data/track1_stanford_results.json"

    test_docs = os.listdir(test_conll_dir)
    results_dict = OrderedDict()
    for test_doc in test_docs:
        print("Processing {} ...".format(test_doc))
        test_statement_filename = test_conll_dir + test_doc
        result = extract_keywords(core_nlp_jar_path,trained_ner_path,test_statement_filename)
        print(result)
        results_dict[test_doc] = result



    print(results_dict)
    print("Saving Track 1 results..")
    with open(track1_results_json, 'w') as outfile:
        json.dump(results_dict, outfile, indent=4)