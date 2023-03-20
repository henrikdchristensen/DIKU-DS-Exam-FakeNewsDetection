from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split


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

def numCountPerLabel(df: pd.DataFrame):
    keys = df['type'].unique()
    typeDict = {k: 0 for k in keys}
    countDict = {k: 0 for k in keys}

    for type, text in zip(df.type, df.cleaned):
        countDict[type] += text.count("<NUM>")
        typeDict[type] += 1

    perTypeDict = {}
    for k, v in countDict.items():
        perTypeDict[k] = v/typeDict[k]

    print(perTypeDict)
    return perTypeDict


def plotNumCount(dict):
    types = list(dict.keys())
    counts = list(dict.values())
    plt.bar(range(len(numCountDict)), counts, tick_label=types)
    plt.show()
