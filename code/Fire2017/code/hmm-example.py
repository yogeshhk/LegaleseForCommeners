__author__ = 'ssbushi'

# Import the toolkit and tags
import nltk
from nltk.corpus import treebank

# Train data - pretagged
train_data = treebank.tagged_sents()[:3000]

print(train_data[0])

# Import HMM module
from nltk.tag import hmm

# Setup a trainer with default(None) values
# And train with the data
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)

print(tagger)
# Prints the basic data about the tagger

print(tagger.tag("Today is a good day .".split()))

print(tagger.tag("Joe met Joanne in Delhi .".split()))

print(tagger.tag("Chicago is the birthplace of Ginny".split()))

"""
Output in order (Notice some tags are wrong :/):
[('Today', u'NN'), ('is', u'VBZ'), ('a', u'DT'), ('good', u'JJ'), ('day', u'NN'), ('.', u'.')]
[('Joe', u'NNP'), ('met', u'VBD'), ('Joanne', u'NNP'), ('in', u'IN'), ('Delhi', u'NNP'), ('.', u'NNP')]
[('Chicago', u'NNP'), ('is', u'VBZ'), ('the', u'DT'), ('birthplace', u'NNP'), ('of', u'NNP'), ('Ginny', u'NNP')]
"""