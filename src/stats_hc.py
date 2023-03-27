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
    'fake': '#1f77b4',
    'reliable': '#ff7f0e',
    'satire': '#2ca02c',
    'bias': '#d62728',
    'conspiracy': '#9467bd',
    'hate': '#8c564b',
    'state': '#e377c2',
    'clickbait': '#7f7f7f',
    'junksci': '#bcbd22',
    'political': '#17becf',
    'unreliable': '#1f77b4',
    'unknown': 'black',
    'nan': 'black'
}


class Statistics():

    def __init__(self, filename: str):
        self.data = pd.read_csv(filename, index_col=False)
        # Convert text-list-of-strins to list of strings
        self.data["content_cleaned"] = self.data["content_cleaned"].apply(literal_eval)
        self.data["sentence_analysis"] = self.data["sentence_analysis"].apply(literal_eval)
    
    def _sort_frequency(self, text, percentage: bool):
        counter = Counter(text)
        sorted_frequency = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words = [word for word, _ in sorted_frequency]
        measure = [freq for _, freq in sorted_frequency]
        if percentage:
            length = len(text)
            measure = [(freq/length)*100 for freq in measure]
        return words, measure

    def barplot(self, data=None, nwords:int = 25, percentage=True, yminmax: tuple=None, ylabel: str=None, title: str=None, ax=None):
        words, measure = self._sort_frequency(text=data, percentage=percentage)
        # Plot bar plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.bar(words[:nwords], measure[:nwords], color='lightsalmon')
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylim(yminmax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    def boxplot(self, data=None, yminmax: tuple=None, ylabel: str=None, title: str=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        boxprops = dict(linewidth=2, color='red', facecolor='lightsalmon')
        # patch_artist must be True to change color the boxes
        ax.boxplot(data, patch_artist=True, boxprops=boxprops)
        ax.set_ylim(yminmax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
    def plot_combined(self):
        # plot bar and box plot together:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 6))
        self.barplot(data=self.data['content_cleaned'].explode().tolist(), nwords=25, percentage=True, ylabel='% of total words', title='word frequency', ax=ax1)
        self.boxplot(data=self.data['content_cleaned'].apply(len), ylabel='# of words', title='# of words per article', ax=ax2)
        self.boxplot(data=self.data['content_cleaned'].apply(lambda x: x.count("!")), ylabel='# of !', title="# of '!' per article", ax=ax3)
        self.boxplot(data=self.data['content_cleaned'].apply(lambda x: x.count("?")), ylabel='# of ?', title="# of '?' per article", ax=ax4)
        fig.tight_layout()
        plt.show()

    def plot_combined_fake_vs_real(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 6))
        words_min = self.data['content_cleaned'].apply(len).min()
        words_max = self.data['content_cleaned'].apply(len).max()
        self.barplot(data=self.data[self.data["type_binary"] == True]['content_cleaned'].explode().tolist(), nwords=25, percentage=True, yminmax=(0, 1.5), ylabel='% of real words', title='real word frequency', ax=ax1)
        self.barplot(data=self.data[self.data["type_binary"] == False]['content_cleaned'].explode().tolist(), nwords=25, percentage=True, yminmax=(0, 1.5), ylabel='% of fake words', title='fake word frequency', ax=ax2)
        self.boxplot(data=self.data[self.data["type_binary"] == True]['content_cleaned'].apply(len), yminmax=(words_min, words_max), ylabel='# of real words', title='# of real words per article', ax=ax3)
        self.boxplot(data=self.data[self.data["type_binary"] == False]['content_cleaned'].apply(len), yminmax=(words_min, words_max), ylabel='# of fake words', title='# of fake words per article', ax=ax4)
        fig.tight_layout()
        plt.show()

    def barplot_type_distribution(self, percentage: bool = True):
        types = self.data['type'].explode().tolist()
        types, measure = self._sort_frequency(text=types, percentage=percentage)
        color_list = [TYPE_COLORS[tp] for tp in types]
        plt.barh(types, measure, color=color_list, alpha=0.5)
        plt.xlabel('% of total labels' if percentage else '# of labels')
        plt.title('label distribution')
        #ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        plt.show()

    def barplot_domain_distribution(self, num: int = None, percentage: bool = True):
        types = self.data['domain'].explode().tolist()
        types, measure = self._sort_frequency(text=types, percentage=percentage)
        colors = plt.cm.tab20(np.arange(len(types)))
        plt.barh(types[:num], measure[:num], color=colors, alpha=0.5)
        plt.xlabel('% of total domains' if percentage else '# of domain')
        plt.title('domain distribution' if percentage else 'domain distribution')
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
            ax = percentages.plot(kind='barh', stacked=True, figsize=(10, 8), width=0.6, color=color_list, alpha=0.5)
            ax.set_xlabel('% of label')
        else:
            ax = counts.plot(kind='barh', stacked=True, figsize=(10, 8), width=0.6, color=color_list, alpha=0.5)
            ax.set_xlabel('# of label')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
        ax.set_title(
            f'domain contribution to label ( ≥ {threshold}%)' if percentage else f'domain contribution to label ( {threshold} most frequent)')
        plt.tight_layout()
        plt.show()


    def plot_combined_sentence_analysis(self):
        # plot bar and box plot together:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))
        polarity_min = self.data['sentence_analysis'].apply(lambda x: x[0]).min()-0.1
        polarity_max = self.data['sentence_analysis'].apply(lambda x: x[0]).max()+0.1
        subjective_min = self.data['sentence_analysis'].apply(lambda x: x[1]).min()-0.1
        subjective_max = self.data['sentence_analysis'].apply(lambda x: x[1]).max()+0.1
        self.boxplot(data=self.data[self.data["type_binary"] == True]['sentence_analysis'].apply(lambda x: x[0]), yminmax=(polarity_min, polarity_max), ylabel='polarity score', title='polarity score for real', ax=ax1)
        self.boxplot(data=self.data[self.data["type_binary"] == False]['sentence_analysis'].apply(lambda x: x[0]), yminmax=(polarity_min, polarity_max), ylabel='polarity score', title='polarity score for fake', ax=ax2)
        self.boxplot(data=self.data[self.data["type_binary"] == True]['sentence_analysis'].apply(lambda x: x[1]), yminmax=(subjective_min, subjective_max), ylabel='subjective score', title='subjective score for real', ax=ax3)
        self.boxplot(data=self.data[self.data["type_binary"] == False]['sentence_analysis'].apply(lambda x: x[1]), yminmax=(subjective_min, subjective_max), ylabel='subjective score', title='subjective score for fake', ax=ax4)
        fig.tight_layout()
        plt.show()