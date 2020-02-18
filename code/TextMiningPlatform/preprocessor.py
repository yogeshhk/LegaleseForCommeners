#======================================================================================================================
#   Takes text corpus dictionary and cleans it up, along with tokenization
#
#   Copyright (C) 2017 Yogesh H Kulkarni
#======================================================================================================================
import string
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

def clean_and_tokenize(text, text_type):
    tokens = []
    if text_type == "text":
        tokens = clean_and_tokenize_sentences(text)
    elif text_type == "tweet":
        tokens = clean_and_tokenize_messages(text)
    return tokens


def clean_and_tokenize_sentences(sentence):
    try:
        # tweet = tweet.decode('utf-8').lower()
        sentence = sentence.lower()
        sentence = re.sub('\s+\s+', '', sentence)  # Remove multiple whitespace
        tokens = tokenizer.tokenize(sentence)
        tokens = [t for t in tokens if t.isalpha()]
        lemmatizer = WordNetLemmatizer()
        stops = set(stopwords.words('english'))  # nltk stopwords list
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stops and len(t)> 2 ]
        if len(tokens) < 3:
            return "NC"
        return tokens
    except:
        return 'NC'


def clean_and_tokenize_messages(tweet):
    try:
        # tweet = tweet.decode('utf-8').lower()
        tweet = tweet.lower()
        tweet = re.sub('https?:\/\/.*\/\w*', '', tweet)  # Remove hyperlinks
        tweet = re.sub('#', '', tweet)  # Remove hashtags
        tweet = re.sub('\@\w*', '', tweet)  # Remove citations
        tweet = re.sub('\$\w*', '', tweet)  # Remove tickers
        tweet = re.sub('[' + string.punctuation + ']+', '', tweet)  # Remove hyperlinks
        tweet = re.sub('\&*[amp]*\;|gt+', '', tweet)  # Remove quotes
        tweet = re.sub('\s+rt\s+', '', tweet)  # Remove retweet
        tweet = re.sub('[\n\t\r]+', '', tweet)  # Remove breaks
        tweet = re.sub('via+\s', '', tweet)  # Remove via
        tweet = re.sub('\s+\s+', '', tweet)  # Remove multiple whitespace

        tokens = tokenizer.tokenize(tweet)
        tokens = [t for t in tokens if t.isalpha()]

        if len(tokens) < 3:
            return "NC"

        return tokens
    except:
        return 'NC'

if __name__ == "__main__":

    message1 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
    message2 = '@remy: This is waaaaayyyy too much for you!'
    message3 = 'Mary had a little lamb. Its fleece was white as snow.'

    clean_message1 = clean_and_tokenize(message1,"tweet")
    clean_message2 = clean_and_tokenize(message2, "tweet")
    clean_message3 = clean_and_tokenize(message3, "text")

    print(clean_message1)
    print(clean_message2)
    print(clean_message3)
