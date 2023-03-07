from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split


def clean_text(df: pd.DataFrame):
    # List of patterns and their appropriate replacements
    patterns = {
        r'(\s{2,})': ' ',
        r'(\t+)': '(\t)',
        r'(\n+)': '(\n)',
        r'(\!|\[|\])': '',
        r'(\d{1,2}[-/\\]\d{1,2}[-/\\]\d{2,4}|\d{2,4}[-/\\]\d{1,2}[-/\\]\d{1,2})|\w{3}\s\d{1,2}\S\d{4}|\d{1,2}\s\w{3}\s\d{4}|(?:jan(?:uary)?|feb(?:ruary)|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?),? \d{2,4},? \d{2,4}|\d{2,4},? (?:jan(?:uary)?|feb(?:ruary)|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?),? \d{2,4}': ' <DATE> ',
        r'([\w.\-]+@(?:[\w-]+\.)+[\w-]{2,4})': '<EMAIL>',
        r'((https?:\/\/)?(?:www\.)?[a-zA-Z0-9-_\+=.:~@#%]+\.[a-zA-Z0-9()]{1,6}\b(?:[a-zA-Z0-9-_.:\\/@#$%&()=+~?]*))': '<URL>',
        r'(\d+)': '<NUM>',
        r'(\.|\,|\?|\–|\&|\—|\”|\“|\%|\:|\-)': ''
    }

    # TODO: Check if escaping special characters is needed.
    # Convert all to text and lowercase all characters
    df = df.applymap(lambda x: str(x).lower())

    # Loop through each pattern and apply the pattern to each row and do replacement if needed
    for pattern, replacement in patterns.items():
        df = df.applymap(lambda x: re.sub(pattern, replacement, x))

    return df


def tokenize_text(df: pd.DataFrame):
    return df.applymap(lambda x: x.split())


def remove_stopwords(df: pd.DataFrame, stopwords: list[str]):
    df = df.applymap(lambda x: ' '.join(
        [word for word in str(x).split() if word not in stopwords]))


def stem(words: list[str]):
    ps = PorterStemmer()
    stemmed_words = []
    for w in words:
        stemmed_words.append(ps.stem(w))
    return stemmed_words


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


class Exploration:

    def countItems(df: pd.DataFrame):
        # make list of tuples with the count of each
        # map =  df.applymap(lambda x: x.count('<URL>'))
        # return map.sum(axis=1)
        urls = 0
        dates = 0
        numbers = 0

        for text in df['content']:
            urls += text.count("<URL>")
            dates += text.count("<DATE>")
            numbers += text.count("<NUM>")

        return {
            "Urls": urls,
            "Dates": dates,
            "Numbers": numbers
        }

    """ 
    Distribution of sources: 
    A bar chart showing the frequency of fake and real 
    news articles from different sources can be used to 
    visualize the distribution of sources. A box plot can also be used to show the spread of the sources' reputability scores.
    """

    def sourceDistribution(df: pd.DataFrame):

        keys = df['type'].unique()
        print(keys)
        # return

        # print(keys)

        # make dict of sources and fake label count
        sourceDict = {}
        typeDict = {k: 0 for k in keys}

        for index, row in df.iterrows():
            print(row['domain'])
            # print(row['unreliable'])

            typeDict[row['type']] = 0

            if row['domain'] in sourceDict:
                # src = sourceDict[row['domain']]
                sourceDict[row['domain']][row['type']] += 1

                # sourceDict[row['domain']] += row['unreliable']
            else:
                # make dict from keys and count
                sourceDict[row['domain']] = {k: 0 for k in keys}
        return sourceDict

    def get_words(content):
        """This regex command should select all words"""
        words = re.findall("(?:\w+[’'.-]?)+", content)
        return words

    def get_word_info(df: pd.DataFrame, column_name='content'):
        """Returns dict witht the etries:  word:(count, frequency)"""
        total_word_count = {}
        for index, row in df.iterrows():
            word_count = Counter(row[column_name])
            for word, count in word_count.items():
                if word in total_word_count:
                    total_word_count[word] += count
                else:
                    total_word_count[word] = count

        total_words = sum(total_word_count.values())
        return {w: (count, count / total_words) for w, count in sorted(total_word_count.items(), key=lambda x: x[1], reverse=True)}

    def plot_word_freq(df: pd.DataFrame, num_words=10000):
        word_info = Exploration.get_word_info(df)
        frequencies = [info[1] for _, info in word_info.items()]
        plt.plot(frequencies)
        plt.xlim([0, len(word_info)])

    def get_stopwords(df: pd.DataFrame, freq_low, freq_high, print_stopwords_info=False):
        stopwords = []
        for k, v in Exploration.get_word_info(df).items():
            if v[1] < freq_low or v[1] > freq_high:
                if print_stopwords_info:
                    print(f"{k}: {v[0]}, {v[1]}")
                stopwords.append(k)
        return stopwords


""" 
Task 4: Split the resulting dataset into a training, validation, and test splits. A common strategy is to uniformly at random split the data 80% / 10% / 10%. You will use the training data to train your baseline and advanced models, the validation data can be used for model selection and hyperparameter tuning, while the test data should only be used in Part 4.
"""

# Split the data into training, validation, and test sets
# A common strategy is to uniformly at random split the data 80% / 10% / 10%.

# 80% training, 10% validation, 10% test


def splitDataSet(pd: pd.DataFrame):
    # split the data into training, validation, and test sets
    # A common strategy is to uniformly at random split the data 80% / 10% / 10%.

    # Split the data into 80% training and 20% validation + test
    train_val, test = train_test_split(df, test_size=0.2, random_state=42)
    # Split the validation data from the remaining 20% of data
    train, val = train_test_split(train_val, test_size=0.5, random_state=42)

    return train, val, test
