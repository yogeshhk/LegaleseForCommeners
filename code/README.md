# Text Mining Platform
(लेख-हन्स extracting from text like a swan नीर-क्षीर विवेक)
Set of scripts to extract relevant information from text corpus
Copyright (C) 2017 Yogesh H Kulkarni

## License:
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or any later version.

## Scripts:
* src/config_engine.py: Extracting certain important fields like name, email, phone, etc. specified in a templates/resume_config.xml like file
** resume_config.xml specifies fields to extract along with the patterns to look for those fields, respectively.
** As all the domain (resume, here) is specified in the config file, any changes or addition to logic is done only the the config file. 
** (Ideally) parser logic is independent and thus should not need any change. Atleast thats the idea!!
** Certain field extraction methods are specified in the config file which are used to parse based on the pattern specified.
* src/reader.py: Collection of various methods, like reading text from URL (), local directory (), etc.
** HTML cleanup to convert to plain text
** Importing text from pdf
* src/preprocessor.py: Cleans raw text corpus, tokenization, then send list of list of tokens per sentence, in a dict with doc name as key

## Dependencies:
* Needs Python 3.5

## ToDos
* Add OCR tesseract, pdf miner capabilities in src/reader.py
* Add automated testing regression suite using 'nose'
* Add word2vec, bow, tfidf, w2v+tfidf vectorization methods
* Add gensim topic modeling, summarization
* Extraction: People names, phone, email, geo locations, organizations, currency, quantitites, events, relations
* Normalize: expand short forms, all lower, unify to single format (phones, address etc)
* Open Source contribution: porting Pattern 2 to Pattern 3. Replace sgml parser w all methods coded

## Study
* 'Pattern': Its has good API importers for Google, Facebook Twitter, etc
* 'Iepy': Information extraction
* 'Spacy': Ready Pos, NER, customization posssible. Good integreation with tensorflow algos
* 'Textpipeliner': https://github.com/krzysiekfonal/textpipeliner
* 'Apache UIMA': Annotation rules in a separate file
* 'Apache Tika': Content analysis. Imports various formats and gets text and image

## Disclaimer:
* Author (yogeshkulkarni@yahoo.com) gives no guarantee of the results of the program. It is just a fun script. Lot of improvements are still to be made. So, don’t depend on it at all.
