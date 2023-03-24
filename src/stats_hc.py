import matplotlib.pyplot as plt
from collections import Counter
from matplotlib_venn import venn2
import pipeline as pp


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