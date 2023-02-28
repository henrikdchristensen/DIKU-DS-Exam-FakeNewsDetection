from nltk import word_tokenize
from nltk.corpus import names, stopwords, words
from nltk.stem import PorterStemmer
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split


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

""" 
1. counting the number of URLs in the content
2. counting the number of dates in the content
3. counting the number of numeric values in the content
4. determining the 100 more frequent words that appear in the content
5. plot the frequency of the 10000 most frequent words (any interesting patterns?)
6. run the analysis in point 4 and 5 both before and after removing stopwords and applying stemming: do you see any difference?
"""






class Preprocessing:

    def countURLs(df: pd.DataFrame):
        # make list of tuples with the count of each
        return df.applymap(lambda x: x.count('<URL>'))

    def countDates(df: pd.DataFrame):
        return df.applymap(lambda x: x.count('<DATE>'))

    def countNumbers(df: pd.DataFrame):
        return df.applymap(lambda x: x.count('<NUM>'))

    def get_words(content):
        """This regex command should select all words"""
        return re.findall("(?:\w+[’'.-]?)+", content)

    def get_word_count(self, df: pd.DataFrame):
        total_word_count = {}
        for index, row in df.iterrows():
            word_count = Counter(self.get_words(row['content']))
            for word, count in word_count.items():
                if word in total_word_count:
                    total_word_count[word] += count
                else:
                    total_word_count[word] = count
                    
        return {k: v for k, v in sorted(total_word_count.items(), key=lambda x: x[1], reverse=True)}

    def get_word_freq(self, df: pd.DataFrame):
        word_count = self.get_word_count(df)
        total_words = sum(word_count.values())
        return ({k: (v / total_words) for k, v in word_count.items()})

    def plot_word_freq(self, df: pd.DataFrame, num_words = 10000):
        frequencies = self.get_word_freq(df)
        plt.plot(frequencies.values())
        plt.xlim([0, num_words])


""" 
Task 4: Split the resulting dataset into a training, validation, and test splits. A common strategy is to uniformly at random split the data 80% / 10% / 10%. You will use the training data to train your baseline and advanced models, the validation data can be used for model selection and hyperparameter tuning, while the test data should only be used in Part 4.
"""

# Split the data into training, validation, and test sets
# A common strategy is to uniformly at random split the data 80% / 10% / 10%.

# 80% training, 10% validation, 10% test

def splitDataSet(pd: pd.DataFrame ): 
    # split the data into training, validation, and test sets
    # A common strategy is to uniformly at random split the data 80% / 10% / 10%.

    # Split the data into 80% training and 20% validation + test
    train_val, test = train_test_split(df, test_size=0.2, random_state=42)
    # Split the validation data from the remaining 20% of data
    train, val = train_test_split(train_val, test_size=0.5, random_state=42)

    return train, val, test

