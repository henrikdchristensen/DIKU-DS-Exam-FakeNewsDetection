import matplotlib.pyplot as plt
from collections import Counter
import pipeline as pp
import numpy as np
import pandas as pd
from ast import literal_eval
from textblob import TextBlob


RAW_DATA = '../datasets/sample/dataset.csv'
CLEANED_DATA = '../datasets/sample/news_sample_cleaned.csv'
CLEANED_DATA_NUM = '../datasets/sample/news_sample_cleaned_num_100k.csv'

# Pre-defined colors for each type
TYPE_COLORS = {
    'fake': 'red',
    'reliable': 'blue',
    'satire': 'pink',
    'bias': 'orange',
    'conspiracy': 'purple',
    'hate': 'green',
    'state': 'brown',
    'clickbait': 'grey',
    'junksci': 'yellow',
    'political': 'magenta',
    'unreliable': 'cyan',
    'unknown': 'black',
    'nan': 'black'
}


class Statistics():

    def __init__(self, filename: str):
        self.data = pd.read_csv(filename, index_col=False)
        # Convert text-list-of-strins to list of strings
        self.data["content"] = self.data["content"].apply(literal_eval)
        self.data['TextBlob_Polarity'] = self.data['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        self.data['TextBlob_Subjectivity'] = self.data['content'].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity)
        self.data["content"] = self.data["content"].apply(lambda x: " ".join(x))
        self.data["content"] = self.data["content"].apply(lambda x: x.split())

    def _sort_frequency(self, text, percentage: bool):
        counter = Counter(text)
        sorted_frequency = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words = [word for word, _ in sorted_frequency]
        measure = [freq for _, freq in sorted_frequency]
        if percentage:
            length = len(text)
            measure = [(freq/length)*100 for freq in measure]
        return words, measure

    def _barplot_word_frequency(self, nwords: int = 25, percentage: bool = True):
        words_list = self.data['content'].explode().tolist()
        words, measure = self._sort_frequency(text=words_list, percentage=percentage)
        fig, ax = plt.subplots()
        ax.bar(words[:nwords], measure[:nwords], color='red', alpha=0.5)
        ax.set_ylabel('% of total words' if percentage else '# of words')
        ax.set_title(
            f'Percentage of total words ({nwords} most frequent)' if percentage else f'Frequency of the {nwords} most frequent words')
        ax.tick_params(axis='x', rotation=90)
        return fig

    def _boxplot_word_frequency(self):
        word_counts = [len(article) for article in self.data['content']]
        boxprops = dict(linewidth=2, color='red', facecolor='lightsalmon')
        fig, ax = plt.subplots()
        # patch_artist must be True to change color the boxes
        ax.boxplot(word_counts, patch_artist=True, boxprops=boxprops)
        ax.set_ylabel('# of words')
        ax.set_title('# of words per article')
        return fig

    def plot_word_frequency(self):
        # plot bar and box plot together:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1 = self._barplot_word_frequency(nwords=25, percentage=True)
        ax2 = self._boxplot_word_frequency()
        fig.tight_layout()
        fig.show()

    def barplot_word_frequency_fake_vs_real(self, nwords: int = 25, binary_label: str = "binary_label", percentage: bool = True):
        real_words_list = self.data[self.data[binary_label] == True]['content'].explode().tolist()
        fake_words_list = self.data[self.data[binary_label] == False]['content'].explode().tolist()
        real_words, real_meausre = self._sort_frequency(text=real_words_list, percentage=percentage)
        fake_words, fake_meausre = self._sort_frequency(text=fake_words_list, percentage=percentage)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('Real')
        ax2.set_title('Fake')
        ax1.bar(real_words[:nwords], real_meausre[:nwords], color='red', alpha=0.5, label="Real")
        ax2.bar(fake_words[:nwords], fake_meausre[:nwords], color='blue', alpha=0.5, label="Fake")
        ax1.set_ylabel('% of words' if percentage else '# of words')
        ax2.set_ylabel('% of words' if percentage else '# of words')
        plt.setp(ax1.get_xticklabels(), rotation=90)
        plt.setp(ax2.get_xticklabels(), rotation=90)
        plt.suptitle('# of words per article - fake vs real')
        plt.tight_layout()
        plt.show()

    def boxplot_word_frequency_fake_vs_real(self, binary_label: str = "binary_label"):
        real_words_list = self.data[self.data[binary_label] == True]['content']
        fake_words_list = self.data[self.data[binary_label] == False]['content']
        real_words_counts = [len(article) for article in real_words_list]
        fake_words_counts = [len(article) for article in fake_words_list]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        real_boxprops = dict(linewidth=2, color='red', facecolor='lightsalmon')
        fake_boxprops = dict(linewidth=2, color='blue', facecolor='lightblue')
        ax1.boxplot(real_words_counts, patch_artist=True, boxprops=real_boxprops)
        ax2.boxplot(fake_words_counts, patch_artist=True, boxprops=fake_boxprops)
        ax1.set_title('Real')
        ax2.set_title('Fake')
        ax1.set_ylabel('# of words')
        ax2.set_ylabel('# of words')
        plt.suptitle('# of words per article')
        plt.tight_layout()
        plt.show()

    def barplot_type_distribution(self, percentage: bool = True):
        types = self.data['type'].explode().tolist()
        types, measure = self._sort_frequency(text=types, percentage=percentage)
        color_list = [TYPE_COLORS[tp] for tp in types]
        plt.barh(types, measure, color=color_list, alpha=0.6)
        plt.xlabel('% of total labels' if percentage else '# of labels')
        plt.title('Percentage of total labels' if percentage else 'Frequency of the labels')
        plt.tight_layout()
        plt.show()

    def barplot_domain_distribution(self, percentage: bool = True):
        types = self.data['domain'].explode().tolist()
        types, measure = self._sort_frequency(text=types, percentage=percentage)
        colors = plt.cm.tab20(np.arange(len(types)))
        plt.barh(types, measure, color=colors, alpha=0.6)
        plt.xlabel('% of domain' if percentage else '# of domain')
        plt.title('Domain distribution')
        plt.tight_layout()
        plt.show()

    def barplot_domain_contribution(self, threshold: float = 0, percentage: bool = True):
        counts = self.data.groupby(['domain', 'type'])['content'].count().unstack()
        percentages = counts.apply(lambda x: x / x.sum() * 100)
        percentages = percentages[percentages > threshold]
        counts = counts[counts > threshold]
        for type in list(percentages.columns):
            percentages.sort_values(type, na_position='first', ascending=False, inplace=True)
        color_list = [TYPE_COLORS[tp] for tp in percentages.columns]
        if percentage:
            ax = percentages.plot(kind='barh', stacked=True, figsize=(10, 8), width=0.6, color=color_list, alpha=0.6)
            ax.set_xlabel('% of articles')
        else:
            ax = counts.plot(kind='barh', stacked=True, figsize=(10, 8), width=0.6, color=color_list, alpha=0.6)
            ax.set_xlabel('# of articles')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
        ax.set_title(
            f'Domain contribution to label ( ≥ {threshold}%)' if percentage else f'Domain contribution to label ( {threshold} most frequent)')
        plt.tight_layout()
        plt.show()

    def boxplot_polarity(self, binary_label: str = "binary_label"):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.boxplot(self.data[self.data[binary_label] == True]['TextBlob_Polarity'])
        ax2.boxplot(self.data[self.data[binary_label] == False]['TextBlob_Polarity'])
        ax1.set_xlabel('Real')
        ax2.set_xlabel('Fake')
        ax1.set_ylabel('# of articles')
        ax2.set_ylabel('# of articles')
        plt.suptitle('Polarity distribution')
        plt.tight_layout()
        plt.show()
        print(f"Median of Polarity real: {self.data[self.data[binary_label] == True]['TextBlob_Polarity'].median()}")
        print(f"Median of Polarity fake: {self.data[self.data[binary_label] == False]['TextBlob_Polarity'].median()}")

    def boxplot_subjectivity(self, binary_label: str = "binary_label"):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.boxplot(self.data[self.data[binary_label] == True]['TextBlob_Subjectivity'])
        ax2.boxplot(self.data[self.data[binary_label] == False]['TextBlob_Subjectivity'])
        ax1.set_xlabel('Real')
        ax2.set_xlabel('Fake')
        ax1.set_ylabel('# of articles')
        ax2.set_ylabel('# of articles')
        plt.suptitle('Subjectivity distribution')
        plt.tight_layout()
        plt.show()
        print(
            f"Median of Subjectivity real: {self.data[self.data[binary_label] == True]['TextBlob_Subjectivity'].median()}")
        print(
            f"Median of Subjectivity fake: {self.data[self.data[binary_label] == False]['TextBlob_Subjectivity'].median()}")
