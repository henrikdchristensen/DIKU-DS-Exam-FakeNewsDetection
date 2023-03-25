import matplotlib.pyplot as plt
from collections import Counter
from matplotlib_venn import venn2
import pipeline as pp
import numpy as np
import pandas as pd


RAW_DATA = '../datasets/sample/dataset.csv'
CLEANED_DATA = '../datasets/sample/news_sample_cleaned.csv'
CLEANED_DATA_NUM = '../datasets/sample/news_sample_cleaned_num_100k.csv'


class Word_frequency(pp.FunctionApplier):
    def __init__(self, nwords=50, labels=('reliable', 'fake')):
        self.nwords = nwords
        self.word_list = []
        self.word_counter = Counter()
        self.label_list = labels
        self.label_to_word_counter = {label: Counter() for label in labels}

    def function_to_apply(self, row):
        word = row['content']
        label = row['type']
        self.word_list.append(word)
        self.word_counter.update(word)
        if label in self.label_list:
            self.label_to_word_counter[label].update(word)
        return row

    def _sort_frequency(self, counter):
        sorted_frequency = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words = [x[0] for x in sorted_frequency[:self.nwords]]
        frequency = [x[1] for x in sorted_frequency[:self.nwords]]
        return words, frequency

    def plot_frequency(self, label=None):
        words, frequency = self._sort_frequency(self.word_counter) if label is None else self._sort_frequency(self.label_to_word_counter[label])
        plt.bar(words, frequency, color='magenta')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of the {self.nwords} most frequent words')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
    def plot_fake_real(self, labels: tuple[str, str] = ("fake", "reliable") ):
        words1, frequency1 = self._sort_frequency(self.label_to_word_counter[labels[0]])
        words2, frequency2 = self._sort_frequency(self.label_to_word_counter[labels[1]])
        # map the word frequency from the fake news word list to the words from the real news
        for i in range(len(words1)):
            if not words1[i] in words2:
                frequency2[i] = 0
        # Set the width of the bars
        bar_width = 0.35
        # Set the positions of the bars on the x-axis
        words1_pos = np.arange(len(words1))
        words2_pos = words1_pos + bar_width
        # Create the figure and axis objects
        fig, ax = plt.subplots()
        # Plot the bars for fake news
        ax.bar(words1_pos, frequency1, width=bar_width, color='b', label=labels[0])
        # Plot the bars for reliable news
        ax.bar(words2_pos, frequency2, width=bar_width, color='g', label=labels[1])
        # Add labels and title to the plot
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_xticks(words1_pos + bar_width / 2)
        ax.set_xticklabels(words1)
        ax.set_title('Word Frequency Comparison')
        # rotate the xticks
        plt.xticks(rotation=90)
        # Add a legend to the plot
        ax.legend()
        # Show the plot
        plt.show()
        
        
class Count_items(pp.FunctionApplier):
    def __init__(self):
        self.numbers = 0
        self.urls = 0
        self.email = 0
        self.dates = 0

    def function_to_apply(self, content):
        for text in content:
            self.numbers += text.count("<number>")
            self.urls += text.count("<url>")
            self.email += text.count("<email>")
            self.dates += text.count("<date>")
        return content
    
    
class Contribution(pp.FunctionApplier): 
    def __init__(self):
        self.data = pd.DataFrame()

    def function_to_apply(self, row):
        self.data = self.data.append(row, ignore_index=True)
        return row

    def contributionPlot(self, threshold=1):
        # group the articles by domain and category, and count the number of articles in each group
        counts = self.data.groupby(['domain', 'type'])['content'].count().unstack()
        # convert the counts to percentages and round to two decimal places
        percentages = counts.apply(lambda x: x / x.sum() * 100).round(2)
        # filter the percentages to only show the contributions above the threshold
        percentages = percentages[percentages > threshold]
        # drop the rows with all NaN values
        percentages = percentages.dropna(how='all')
        # rearrange the rows so that the order of the categories is consistent
        for type in list(percentages.columns):
            percentages.sort_values(type, na_position='first', ascending=False, inplace=True)
        # create a stacked horizontal bar chart of the percentages
        ax = percentages.plot(kind='barh', stacked=True, figsize=(10, 8), width=0.6)
        # set the x-axis label to show the percentages
        ax.set_xlabel('Percentage')
        # set the legend to display outside the chart
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        # make fontsize smaller of the domain labels
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
        title = f'Contribution of Domains to Categories ( ≥ {threshold}%)'
        ax.set_title(title)
        # show the chart
        plt.show()


# make class to count each type fake, real, unreliable, reliable etc. and make a frequency plot
class Article_Type_frequency(pp.FunctionApplier):
    # kg = 20
    def __init__(self):
        self.frequency = Counter()
        self.sorted_frequency = []
        self.items = 13

    def function_to_apply(self, type):
        # Update/add list of word
        type = str(type)
        # print(self.frequency)
        self.frequency.update({type: 1})
        # Return the sorted dictionary based on the frequency of each word
        self.sorted_frequency = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        # print("sorted_frequency", self.sorted_frequency)
        return type

    def plot(self):
        # print(self.sorted_frequency)
        # Extract the words and their frequency from the sorted list
        words = [x[0] for x in self.sorted_frequency[:self.items]]
        frequency = [x[1] for x in self.sorted_frequency[:self.items]]
        # Plot a barplot using matplotlib
        plt.bar(words, frequency)
        plt.ylabel('Frequency')
        plt.title(f'Frequency of the {self.items} most frequent words')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plotDistribution(self):
        # Extract the words and their frequency from the sorted list
        words = [x[0] for x in self.sorted_frequency]
        frequency = [x[1] for x in self.sorted_frequency]

        total_frequency = sum(frequency)

        # Compute the probability of each word.
        probabilities = [x / total_frequency for x in frequency]

        # sort the words by their probability
        # sorted_probability = sorted(probability, reverse=True)
        # Sort the words based on their probability.
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probabilities = [probabilities[i] for i in sorted_indices]

        # Compute the cumulative probability of each word.
        cumulative_probabilities = np.cumsum(sorted_probabilities)

        plt.bar(words, sorted_probabilities)
        plt.xlabel('Type')
        plt.ylabel('Frequency')
        plt.title('Type Frequency Distribution')
        plt.show()