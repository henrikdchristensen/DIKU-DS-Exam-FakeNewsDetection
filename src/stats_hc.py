import matplotlib.pyplot as plt
from collections import Counter
from matplotlib_venn import venn2
import pipeline as pp
import numpy as np
import pandas as pd
from ast import literal_eval


RAW_DATA = '../datasets/sample/dataset.csv'
CLEANED_DATA = '../datasets/sample/news_sample_cleaned.csv'
CLEANED_DATA_NUM = '../datasets/sample/news_sample_cleaned_num_100k.csv'



class Statistics():
    def __init__(self, filename: str):
        self.data = pd.read_csv(filename, index_col=False).iloc[:100]
            
    def _sort_frequency(self, words_list, percentage: bool):
        length = len(words_list)
        counter = Counter(words_list)
        sorted_frequency = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words = [word for word, _ in sorted_frequency]
        measure = [freq for _, freq in sorted_frequency]
        if percentage: # calculate percentage
            measure = [(freq/length)*100 for freq in measure]
        return words, measure
    
    def plot_frequency(self, column: str, nwords: int = 25, percentage: bool = False):
        # Get one list of strings from the column where every row is a list of strings:
        self.data["content"] = self.data["content"].apply(literal_eval)
        words_list = self.data[column].explode().tolist()
        words, measure = self._sort_frequency(words_list=words_list, percentage=percentage)
        if percentage:
            plt.bar(words[:nwords], measure[:nwords], color='red', alpha=0.5)
            plt.ylabel('% of total words')
            plt.title(f'Percentage of total words ({nwords} most frequent)')
        else:
            plt.bar(words[:nwords], measure[:nwords], color='red', alpha=0.5)
            plt.ylabel('# of words')
            plt.title(f'Frequency of the {nwords} most frequent words')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
    def plot_fake_real(self, nwords: int = 25, binary_label: str = "binary_label", percentage: bool = False):
        # get the words and frequency for each label
        real_words_list = self.data[self.data[binary_label] == True]['content'].explode().tolist()
        fake_words_list = self.data[self.data[binary_label] == False]['content'].explode().tolist()
        # sort the words and frequency
        real_words, real_meausre = self._sort_frequency(words_list=real_words_list, percentage=percentage)
        fake_words, fake_meausre = self._sort_frequency(words_list=fake_words_list, percentage=percentage)
        # create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title(f'Real')
        ax2.set_title(f'Fake')
        # calculate percentage for each label
        ax1.bar(real_words[:nwords], real_meausre[:nwords], color='red', alpha=0.5, label="Real")
        ax2.bar(fake_words[:nwords], fake_meausre[:nwords], color='blue', alpha=0.5, label="Fake")
        if percentage:
            ax1.set_ylabel('% of words')
            ax2.set_ylabel('% of words')
        else:
            ax1.set_ylabel('# of words')
            ax2.set_ylabel('# of words')
        # rotate the xticks
        plt.setp(ax1.get_xticklabels(), rotation=90)
        plt.setp(ax2.get_xticklabels(), rotation=90)
        plt.tight_layout()
        plt.show()
    
    def plot_domain_contribution(self, threshold=1):
        # group the articles by domain and category, and count the number of articles in each group
        counts = self.data.groupby(['domain', 'type'])['content'].count().unstack()
        # convert the counts to percentages and round to two decimal places
        percentages = counts.apply(lambda x: x / x.sum() * 100)
        # filter the percentages to only show the contributions above the threshold
        percentages = percentages[percentages > threshold]
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
        plt.tight_layout()
        plt.show()
