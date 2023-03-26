import matplotlib.pyplot as plt
from collections import Counter
from matplotlib_venn import venn2
import pipeline as pp
import numpy as np
import pandas as pd
from ast import literal_eval
import seaborn as sns


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
}


class Statistics():
    
    def __init__(self, filename: str):
        self.data = pd.read_csv(filename, index_col=False).iloc[:100]
        # Convert text-list-of-strins to list of strings 
        self.data["content"] = self.data["content"].apply(literal_eval)
        
    def _sort_frequency(self, text, percentage: bool):
        counter = Counter(text)
        sorted_frequency = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words = [word for word, _ in sorted_frequency]
        measure = [freq for _, freq in sorted_frequency]
        if percentage:
            length = len(text)
            measure = [(freq/length)*100 for freq in measure]
        return words, measure
    
    def barplot_word_frequency(self, nwords: int = 25, percentage: bool = False):
        words_list = self.data['content'].explode().tolist()
        words, measure = self._sort_frequency(text=words_list, percentage=percentage)
        plt.bar(words[:nwords], measure[:nwords], color='red', alpha=0.5)
        plt.ylabel('% of total words' if percentage else '# of words')
        plt.title(f'Percentage of total words ({nwords} most frequent)' if percentage else f'Frequency of the {nwords} most frequent words')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    
    def boxplot_word_frequency(self):
        word_counts = [len(article) for article in self.data['content']]
        fig, ax = plt.subplots()
        boxprops = dict(linewidth=2, color='red', facecolor='lightsalmon')
        ax.boxplot(word_counts, patch_artist=True, boxprops=boxprops) # patch_artist must be True to change color the boxes
        ax.set_ylabel('# of words')
        ax.set_title('# of words per article')
        plt.tight_layout()
        plt.show()
        
    def barplot_word_frequency_fake_vs_real(self, nwords: int = 25, binary_label: str = "binary_label", percentage: bool = False):
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
        ax1.boxplot(real_words_counts, patch_artist=True,boxprops=real_boxprops)
        ax2.boxplot(fake_words_counts, patch_artist=True, boxprops=fake_boxprops)
        ax1.set_xlabel('Real')
        ax2.set_xlabel('Fake')
        ax1.set_ylabel('# of words')
        ax2.set_ylabel('# of words')
        plt.suptitle('Fake vs. Real\n# of words per article')
        plt.tight_layout()
        plt.show()
    
    def barplot_domain_contribution(self, threshold: float = 1, percentage: bool = False):
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
        ax.set_title(f'Domain contribution to label ( ≥ {threshold}%)' if percentage else f'Domain contribution to label ( {threshold} most frequent)')
        plt.tight_layout()
        plt.show()
        
    def barplot_type_distribution(self, percentage: bool = False):
        types = self.data['type'].explode().tolist()
        types, measure = self._sort_frequency(text=types, percentage=percentage)
        color_list = [TYPE_COLORS[tp] for tp in types]
        plt.barh(types, measure, color=color_list, alpha=0.6)
        plt.xlabel('% of total labels' if percentage else '# of labels')
        plt.title(f'Percentage of total labels' if percentage else f'Frequency of the labels')
        plt.tight_layout()
        plt.show()
        
    def barplot_authors_contribution(self, threshold: float = 1, percentage: bool = False):
        counts = self.data.groupby(['authors', 'type'])['content'].count().unstack()
        percentages = counts.apply(lambda x: x / x.sum() * 100)
        percentages = percentages[percentages > threshold]
        counts = counts[counts > threshold]
        for type in list(percentages.columns):
            percentages.sort_values(type, na_position='first', ascending=False, inplace=True)
        if percentage:
            ax = percentages.plot(kind='barh', stacked=True, figsize=(10, 8), width=0.6, alpha=0.6)
            ax.set_xlabel('Percentage')
        else:
            ax = counts.plot(kind='barh', stacked=True, figsize=(10, 8), width=0.6, alpha=0.6)
            ax.set_xlabel('# of articles')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
        ax.set_title(f'Author contribution to label ( ≥ {threshold})')
        plt.tight_layout()
        plt.show()
        
    def barplot_authors_distribution(self, percentage: bool = False):
        types = self.data['authors'].explode().tolist()
        types, measure = self._sort_frequency(text=types, percentage=percentage)
        plt.barh(types, measure, alpha=0.6)
        plt.xlabel('% of total labels' if percentage else '# of labels')
        plt.title(f'Percentage of total authors' if percentage else f'Frequency of the authors')
        plt.tight_layout()
        plt.show()