from nltk import tokenize
from nltk.corpus import names, stopwords, words
from nltk.stem import PorterStemmer
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import re


def clean_text(df: pd.DataFrame):
    # List of patterns and their appropriate replacements
    patterns = [
        (r'(\s{2,})', ' '),
        (r'(\t+)', '(\t)'),
        (r'(\n+)', '(\n)'),
        (r'(\d{1,2}[-/\\]\d{1,2}[-/\\]\d{2,4}|\d{2,4}[-/\\]\d{1,2}[-/\\]\d{1,2})|\w{3}\s\d{1,2}[\s,]\d{4}|\d{1,2}\s\w{3}\s\d{4}|(?:jan(?:uary)?|feb(?:ruary)|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[\s,]\d{2}[\s,]\d{2,4}|\d{2}[\s,](?:jan(?:uary)?|feb(?:ruary)|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[\s,]\d{2,4}', ' <DATE> '),
        (r'([\w.\-]+@(?:[\w-]+\.)+[\w-]{2,4})', '<EMAIL>'),
        (r'(https?:\/\/(?:www\.)?[a-zA-Z0-9-_\+=.:~@#%]+\.[a-zA-Z0-9()]{1,6}\b(?:[a-zA-Z0-9-_.:\\/@#$%&()=+~]*))', '<URL>'),
        (r'(\d+)', '<NUM>'),
        (r'(\.|\,|\?|\!|\–|\&|\[|\]|\—|\”|\“)|\%|\:', '\ ')
    ]
    # TODO: Check if escaping special characters is needed.
    # Lowercase all characters
    df = df.applymap(lambda x: str(x).lower())

    # Loop through each pattern and apply the pattern to each column and do replacement if needed
    for reg, replace in patterns:
        df = df.applymap(
            lambda x: re.sub(reg, replace, str(x)))

    return df


def tokenize_text(df: pd.DataFrame):
    return df.applymap(lambda x: word_tokenize(x))


def stem(words: list[str]):
    ps = PorterStemmer()
    for w in words:
        # print(w, " : ", ps.stem(w))
        ps.stem(w)
    return words

 # compute the size of the vocabulary.


# def vocabulary_size(words: list[str]):
# return len(set(words))


def word_frequency(words):
    # Create a Counter list
    frequency = Counter()
    # Update/add list of words
    frequency.update(words)

    # Return the sorted dictionary based on the frequency of each word
    return sorted(frequency.items(), key=lambda x: x[1], reverse=True)


def plot_word_frequency(sorted_frequency):
    # Number of words to list of frequency
    nwords = 50

    # Extract the words and their frequency from the sorted list
    words = [x[0] for x in sorted_frequency[:nwords]]
    frequency = [x[1] for x in sorted_frequency[:nwords]]

    # Plot a barplot using matplotlib
    plt.bar(words, frequency)
    plt.ylabel('Frequency')
    plt.title('Frequency of the 50 most frequent words')
    plt.xticks(rotation=90)
    plt.show()


# Compute the reduction : list[str]rate of the voc: list[str]abulary size after stemming.
# def oldCount = vocabulary_size(words)reduction_rate(words, stemmed_words):


# oldCountvooldCountoldCount return (vocabulary_size(words) - vocabulary_size(stemmed_words)) / vocabulary_size(words)

# kf counting the number of URLs in the content
