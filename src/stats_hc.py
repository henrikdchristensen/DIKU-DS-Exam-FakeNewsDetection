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
        self.first_person_pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        self.second_person_pronouns = ['you', 'your', 'yours']
        self.third_person_pronouns = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
        self.pronouns = self.first_person_pronouns + self.second_person_pronouns + self.third_person_pronouns
        self.negations = ['not', 'never', 'neither', 'nor', 'barely', 'hardly', 'scarcely', 'seldom', 'rarely', 'no', 'neither', 'neither', 'nothing', 'none', 'no one', 'nobody', 'nowhere']
        
    def _average_sentence_lengths(self, words):
        sentence_lengths = []
        current_sentence_length = 0
        for word in words:
            current_sentence_length += 1
            if word in ['.', '!', '?']:
                sentence_lengths.append(current_sentence_length)
                current_sentence_length = 0
        if sentence_lengths:
            return sum(sentence_lengths) / len(sentence_lengths)
        else:
            return 0
                
    def _sort_frequency(self, text, percentage: bool):
        counter = Counter(text)
        sorted_frequency = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words = [word for word, _ in sorted_frequency]
        measure = [freq for _, freq in sorted_frequency]
        if percentage:
            length = len(text)
            measure = [(freq/length)*100 for freq in measure]
        return words, measure

    def _barplot(self, data=None, nwords: int = 25, percentage=True, minmax: tuple = None, label: str = None, title: str = None, boolean=False, color='lightsalmon', ax=None):
        words, measure = self._sort_frequency(text=data, percentage=percentage)
        # Plot bar plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        #ax.tick_params(axis='x', rotation=90)
        if boolean:
            color = ['lightblue', 'lightsalmon']
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['True', 'Fake'])
        ax.barh(words[:nwords], measure[:nwords], color=color)
        ax.set_xlim(minmax)
        ax.set_xlabel(label)
        ax.set_title(title)

    def _boxplot(self, data=None, minmax: tuple = None, label: str = None, title: str = None, color='lightsalmon', ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        boxprops = dict(linewidth=2, facecolor=color)
        # patch_artist must be True to change color the boxes
        ax.boxplot(data, patch_artist=True, boxprops=boxprops)
        ax.set_ylim(minmax)
        ax.set_ylabel(label)
        ax.set_title(title)
    
    def plot_word_frequency_barplot(self):
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        words_min = self.data['content_cleaned'].apply(len).min()
        words_max = self.data['content_cleaned'].apply(len).max()
        self._barplot(data=self.data['content_cleaned'].explode().tolist(), nwords=25, 
                      percentage=True, label='% of total words', title='word frequency', color='yellowgreen', ax=ax1)
    
    def plot_word_frequency_boxplot(self):
        # plot bar and box plot together:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
        self._boxplot(data=self.data['content_cleaned'].apply(len),
                     label='# of words', title='# of words per article', color='yellowgreen', ax=ax1)
        self._boxplot(data=self.data['content_cleaned'].apply(lambda x: x.count("!")),
                     label='# of !', title="# of '!' per article", color='yellowgreen', ax=ax2)
        self._boxplot(data=self.data['content_cleaned'].apply(lambda x: x.count("?")),
                     label='# of ?', title="# of '?' per article", color='yellowgreen', ax=ax3)
        fig.tight_layout()
        plt.show()
    


    def plot_word_frequency_fake_vs_real(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))
        words_min = self.data['content_cleaned'].apply(len).min()
        words_max = self.data['content_cleaned'].apply(len).max()
        self._barplot(data=self.data[self.data["type_binary"] == True]['content_cleaned'].explode().tolist(
        ), nwords=25, percentage=True, minmax=(0, 1.5), label='% of real words', title='real word frequency', color='lightsalmon', ax=ax1)
        self._barplot(data=self.data[self.data["type_binary"] == False]['content_cleaned'].explode().tolist(
        ), nwords=25, percentage=True, minmax=(0, 1.5), label='% of fake words', title='fake word frequency', color='lightblue', ax=ax2)
        self._boxplot(data=self.data[self.data["type_binary"] == True]['content_cleaned'].apply(len), minmax=(
            words_min, words_max), label='# of real words', title='# of real words per article', color='lightsalmon', ax=ax3)
        self._boxplot(data=self.data[self.data["type_binary"] == False]['content_cleaned'].apply(len), minmax=(
            words_min, words_max), label='# of fake words', title='# of fake words per article', color='lightblue', ax=ax4)
        fig.tight_layout()
        plt.show()
        
    def barplot_type(self, percentage: bool = True, ax=None):
        types = self.data['type'].explode().tolist()
        types, measure = self._sort_frequency(text=types, percentage=percentage)
        color_list = [TYPE_COLORS[tp] for tp in types]
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.barh(types, measure, color=color_list, alpha=0.5)
        ax.set_xlabel('% of total labels' if percentage else '# of labels')
        ax.set_title('label distribution')
        #ax.tick_params(axis='x', rotation=90)

    def plot_type_fake_vs_real(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.barplot_type(percentage=True, ax=ax1)
        self._barplot(data=self.data["type_binary"].tolist(),
                     percentage=True, boolean=True, title="real vs. fake distribution", label="% of total labels", ax=ax2)
        fig.tight_layout()
        plt.show()
    
    
    def barplot_domain(self, num: int = None, percentage: bool = True):
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
    
    def plot_average_sentence_length_fake_vs_real(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        true = self.data[self.data["type_binary"] == True]['content_cleaned'].apply(self._average_sentence_lengths)
        fake = self.data[self.data["type_binary"] == False]['content_cleaned'].apply(self._average_sentence_lengths)
        max_val = max(true.max(), fake.max())+2
        self._boxplot(data=true, minmax=(-10, max_val), label='avg. sentence length', title='real avg. sentence length', color='lightsalmon', ax=ax1)
        self._boxplot(data=fake, minmax=(-10, max_val), label='avg. sentence length', title='fake avg. sentence length', color='lightblue', ax=ax2)
        fig.tight_layout()
        plt.show()
    
    def plot_pronouns_fake_vs_real(self):
        # Create 2x4 sub plots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(12, 12))
        
        true_first = self.data[self.data["type_binary"] == True]['content_cleaned'].apply(lambda x: sum([x.count(word) for word in self.first_person_pronouns]))
        fake_first = self.data[self.data["type_binary"] == False]['content_cleaned'].apply(lambda x: sum([x.count(word) for word in self.first_person_pronouns]))
        max_val_first = max(true_first.max(), fake_first.max())+2
        self._boxplot(data=true_first, minmax=(-0.4, max_val_first), label='# of 1st pronouns', title='real 1st person pronouns', color='lightsalmon', ax=ax1)
        self._boxplot(data=fake_first, minmax=(-0.4, max_val_first), label='# of 1st pronouns', title='fake 1st person pronouns', color='lightblue', ax=ax2)
        
        true_second = self.data[self.data["type_binary"] == True]['content_cleaned'].apply(lambda x: sum([x.count(word) for word in self.second_person_pronouns]))
        fake_second = self.data[self.data["type_binary"] == False]['content_cleaned'].apply(lambda x: sum([x.count(word) for word in self.second_person_pronouns]))
        max_val_second = max(true_second.max(), fake_second.max())+2
        self._boxplot(data=true_second, minmax=(-0.4, max_val_second), label='# of 2nd pronouns', title='real 2nd person pronouns', color='lightsalmon', ax=ax3)
        self._boxplot(data=fake_second, minmax=(-0.4, max_val_second), label='# of 2nd pronouns', title='fake 2nd person pronouns', color='lightblue', ax=ax4)
        
        true_third = self.data[self.data["type_binary"] == True]['content_cleaned'].apply(lambda x: sum([x.count(word) for word in self.third_person_pronouns]))
        fake_third = self.data[self.data["type_binary"] == False]['content_cleaned'].apply(lambda x: sum([x.count(word) for word in self.third_person_pronouns]))
        max_val_third = max(true_third.max(), fake_third.max())+2
        self._boxplot(data=true_third, minmax=(-0.4, max_val_third), label='# of 3rd pronouns', title='real 3rd person pronouns', color='lightsalmon', ax=ax5)
        self._boxplot(data=fake_third, minmax=(-0.4, max_val_third), label='# of 3rd pronouns', title='fake 3rd person pronouns', color='lightblue', ax=ax6)
        
        true_total = self.data[self.data["type_binary"] == True]['content_cleaned'].apply(lambda x: sum([x.count(word) for word in self.pronouns]))
        fake_total = self.data[self.data["type_binary"] == False]['content_cleaned'].apply(lambda x: sum([x.count(word) for word in self.pronouns]))
        max_val_total = max(true_total.max(), fake_total.max())+2
        self._boxplot(data=true_total, minmax=(-0.4, max_val_total), label='# of pronouns', title='real pronouns', color='lightsalmon', ax=ax7)
        self._boxplot(data=fake_total, minmax=(-0.4, max_val_total), label='# of pronouns', title='fake pronouns', color='lightblue', ax=ax8)
        fig.tight_layout()
        plt.show()
        
    def plot_negations_fake_vs_real(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        true = self.data[self.data["type_binary"] == True]['content_cleaned'].apply(lambda x: sum([x.count(word) for word in self.negations]))
        fake = self.data[self.data["type_binary"] == False]['content_cleaned'].apply(lambda x: sum([x.count(word) for word in self.negations]))
        max_val = max(true.max(), fake.max())+2
        self._boxplot(data=true, minmax=(-0.4, max_val), label='# of negations', title='real negations', color='lightsalmon', ax=ax1)
        self._boxplot(data=fake, minmax=(-0.4, max_val), label='# of negations', title='fake negations', color='lightblue', ax=ax2)
        fig.tight_layout()
        plt.show()


    def plot_sentence_analysis_fake_vs_real(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))
        polarity_min = self.data['sentence_analysis'].apply(lambda x: x[0]).min()-0.1
        polarity_max = self.data['sentence_analysis'].apply(lambda x: x[0]).max()+0.1
        subjective_min = self.data['sentence_analysis'].apply(lambda x: x[1]).min()-0.1
        subjective_max = self.data['sentence_analysis'].apply(lambda x: x[1]).max()+0.1
        self._boxplot(data=self.data[self.data["type_binary"] == True]['sentence_analysis'].apply(lambda x: x[0]), minmax=(
            polarity_min, polarity_max), label='polarity score', title='polarity score for real', color='lightsalmon', ax=ax1)
        self._boxplot(data=self.data[self.data["type_binary"] == False]['sentence_analysis'].apply(lambda x: x[0]), minmax=(
            polarity_min, polarity_max), label='polarity score', title='polarity score for fake', color='lightblue', ax=ax2)
        self._boxplot(data=self.data[self.data["type_binary"] == True]['sentence_analysis'].apply(lambda x: x[1]), minmax=(
            subjective_min, subjective_max), label='subjective score', title='subjective score for real', color='lightsalmon', ax=ax3)
        self._boxplot(data=self.data[self.data["type_binary"] == False]['sentence_analysis'].apply(lambda x: x[1]), minmax=(
            subjective_min, subjective_max), label='subjective score', title='subjective score for fake', color='lightblue', ax=ax4)
        fig.tight_layout()
        plt.show()
