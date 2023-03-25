import matplotlib.pyplot as plt
from collections import Counter
from matplotlib_venn import venn2
import pipeline as pp
import numpy as np


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

    def _sort_frequency(self, counter):
        sorted_frequency = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words = [x[0] for x in sorted_frequency[:self.nwords]]
        frequency = [x[1] for x in sorted_frequency[:self.nwords]]
        return words, frequency

    def plot_frequency(self, label=None):
        words, frequency = self._sort_frequency(self.word_counter) if label is None else self._sort_frequency(self.label_to_word_counter[label])
        plt.bar(words, frequency)
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