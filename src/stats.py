import matplotlib.pyplot as plt
from collections import Counter
import pipeline as pp
import numpy as np
import pandas as pd
from ast import literal_eval
import pipeline as pp
import filehandling as fh
from matplotlib_venn import venn2


def plot_colors():
    tab20 = plt.get_cmap('tab20').colors
    hex_colors = ['#' + ''.join(f'{int(c*255):02x}' for c in color) for color in tab20]
    print(hex_colors)


class Statistics():
    def __init__(self):
        self.first_pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        self.second_pronouns = ['you', 'your', 'yours']
        self.third_pronouns = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
        self.pronouns = self.first_pronouns + self.second_pronouns + self.third_pronouns
        self.negations = ['not', 'never', 'neither', 'nor', 'barely', 'hardly', 'scarcely',
                          'seldom', 'rarely', 'no', 'neither', 'neither', 'nothing', 'none', 'nobody', 'nowhere']

    def average_sentence_lengths(self, words):
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

    def sort_frequency(self, text, percentage: bool):
        counter = Counter(text)
        sorted_frequency = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words = [word for word, _ in sorted_frequency]
        measure = [freq for _, freq in sorted_frequency]
        if percentage:
            length = len(text)
            measure = [(freq/length)*100 for freq in measure]
        return words, measure

    def barplot(self, data, measure, num: int = 25, minmax: tuple = None, label: str = None, title: str = None, boolean=False, color='lightsalmon', ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        #ax.tick_params(axis='x', rotation=90)
        if boolean:
            color = ['lightblue', 'lightsalmon']
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Fake', 'Reliable'])
        ax.barh(data[:num], measure[:num], color=color, height=0.4)
        ax.set_xlim(minmax)
        ax.set_xlabel(label)
        ax.set_title(title)

    def boxplot_data1_vs_data2(self, data=None, showfliers=True, colors=['lightsalmon', 'lightblue'], ylabel: str = None, title: str = None, ax=None):
        bp = ax.boxplot(data.values(), patch_artist=True, showfliers=showfliers, widths=0.25)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_xticklabels(data.keys())
        ax.set_ylabel(ylabel)
        ax.set_title(title)


class FakeNewsCorpus():
    _initialized = False

    def __init__(self, data: pd.DataFrame, type_label: str = None, binary_type_label: str = None, content_label: str = None, domain_label: str = None, sentence_analysis_label: str = None):
        self.data = data
        self.type_label = type_label
        self.binary_type_label = binary_type_label
        self.content_label = content_label
        self.domain_label = domain_label
        self.sentence_analysis_label = sentence_analysis_label
        # Convert text-list-of-strins to list of strings
        if content_label is not None and not FakeNewsCorpus._initialized:
            self.data[content_label] = self.data[content_label].apply(literal_eval)
        if sentence_analysis_label is not None and not FakeNewsCorpus._initialized:
            self.data[sentence_analysis_label] = self.data[sentence_analysis_label].apply(literal_eval)
        # Colors:
        types = [
            'fake',
            'conspiracy',
            'junksci',
            'hate',
            'unreliable',
            'bias',
            'satire',
            'reliable',
            'clickbait',
            'political',
        ]
        colors = plt.get_cmap('tab10').colors[:len(types)]
        unknown_types = set(types)
        colors += tuple(['black'] * len(unknown_types))
        self.types_colors = {types[i]: colors[i] for i in range(len(types))}
        FakeNewsCorpus._initialized = True


class Statistics_FakeNewsCorpus(Statistics, FakeNewsCorpus):
    def __init__(self, fake_news: FakeNewsCorpus):
        self.stat = Statistics()
        self.fake_news = fake_news

    def barplot_word_frequency(self, nwords=25):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
        true_words, true_words_cnt = self.stat.sort_frequency(
            self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == False][self.fake_news.content_label].explode().tolist(), percentage=True)
        fake_words, fake_words_cnt = self.stat.sort_frequency(
            self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == True][self.fake_news.content_label].explode().tolist(), percentage=True)
        max_value = max(max(true_words_cnt), max(fake_words_cnt))*1.1
        self.stat.barplot(data=true_words, measure=true_words_cnt, num=nwords, minmax=(0, max_value),
                          label='% of words in reliable articles', title='Reliable', color='lightsalmon', ax=ax1)
        self.stat.barplot(data=fake_words, measure=fake_words_cnt, num=nwords, minmax=(0, max_value),
                          label='% of words in fake articles', title='Fake', color='lightblue', ax=ax2)
        fig.tight_layout()
        fig.suptitle('Word frequency', fontsize=16)
        plt.show()

    def boxplot_word_frequency(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))

        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == False][self.fake_news.content_label].apply(len)
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == True][self.fake_news.content_label].apply(len)
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='# of words', ax=ax1, title='With outliers')

        # Without outliers:
        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == False][self.fake_news.content_label].apply(len)
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == True][self.fake_news.content_label].apply(len)
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='# of words', ax=ax2,
                                         showfliers=False, title='Without outliers')

        fig.suptitle('# of words per article', fontsize=16)
        fig.tight_layout()
        plt.show()

    def boxplot_char_frequency(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == False][self.fake_news.content_label].apply(lambda x: x.count("!"))
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == True][self.fake_news.content_label].apply(lambda x: x.count("!"))
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='# of !', title="# of '!' per article", ax=ax1)

        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == False][self.fake_news.content_label].apply(lambda x: x.count("?"))
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == True][self.fake_news.content_label].apply(lambda x: x.count("?"))
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='# of ?', title="# of '?' per article", ax=ax2)

        #fig.suptitle('frequencies', fontsize=16)
        fig.tight_layout()
        plt.show()

    def word_frequency(self):
        print(
            f'\nReliable:\n{self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == False][self.fake_news.content_label].apply(len).describe()}')
        print(
            f'\nFake:\n{self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == True][self.fake_news.content_label].apply(len).describe()}')

    def barplot_type(self, percentage: bool = True, ax=None, title=None):
        types = self.fake_news.data[self.fake_news.type_label].explode().tolist()
        types, measure = self.stat.sort_frequency(text=types, percentage=percentage)
        color_list = [self.fake_news.types_colors[tp] for tp in types]
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.barh(types, measure, color=color_list, alpha=0.5)
        ax.set_xlabel('% of total labels' if percentage else '# of labels')
        ax.set_title(title)
        #ax.tick_params(axis='x', rotation=90)

    def plot_type(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        self.barplot_type(percentage=True, ax=ax1, title="Label distribution")
        words, words_cnt = self.stat.sort_frequency(
            self.fake_news.data[self.fake_news.binary_type_label].tolist(), percentage=True)
        self.stat.barplot(data=words, measure=words_cnt, boolean=True,
                          title="Reliable vs. Fake distribution", label="% of total labels", ax=ax2)

        fig.suptitle('Labels', fontsize=16)
        fig.tight_layout()
        plt.show()

    def barplot_domain(self, num: int = None, percentage: bool = True):
        types = self.fake_news.data[self.fake_news.domain_label].explode().tolist()
        types, measure = self.stat.sort_frequency(text=types, percentage=percentage)
        colors = plt.cm.tab20(np.arange(len(types)))
        fig, ax = plt.subplots(1, 1, figsize=(5, 7))
        ax.barh(types[:num], measure[:num], color=colors, alpha=0.5)
        ax.set_xlabel('% of total domains' if percentage else '# of domain')

        ax.set_title('Domain distribution' if percentage else 'Domain distribution', fontsize=16)
        fig.tight_layout()
        plt.show()

    def barplot_domain_to_label_contribution(self, threshold=0, percentage=True):
        fig, ax = plt.subplots(figsize=(7, 8))

        counts, percentages = None, None
        counts = self.fake_news.data.groupby([self.fake_news.domain_label, self.fake_news.type_label])[
            self.fake_news.content_label].count().unstack()
        # Remove percentages lower than threshold:
        percentages = counts.apply(lambda x: x / x.sum() * 100)
        percentages = percentages[percentages >= threshold]
        percentages = percentages.dropna(how='all')
        for label in list(percentages.columns):
            percentages.sort_values(label, na_position='first', ascending=False, inplace=True)
        color_list = [self.fake_news.types_colors[label] for label in percentages.columns]
        plot_title = f'Domain contribution to Label ( ≥ {threshold}%)' if percentage else f'Domain contribution to Label (top {threshold} most frequent)'
        xlabel = '% of label' if percentage else '# of label'
        if percentage:
            percentages.plot(kind='barh', stacked=True, width=0.5, color=color_list, alpha=0.5, ax=ax)
        else:
            counts[counts >= threshold].plot(kind='barh', stacked=True, width=0.5, color=color_list, alpha=0.5, ax=ax)
        ax.set_xlabel(xlabel)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_title(plot_title, fontsize=16)

        plt.tight_layout()
        plt.show()

    def plot_average_sentence_length(self):
        fig, ax1 = plt.subplots(1, 1, figsize=(3, 4))

        # Plot: average sentence length
        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] ==
                                   False][self.fake_news.content_label].apply(self.stat.average_sentence_lengths)
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] ==
                                   True][self.fake_news.content_label].apply(self.stat.average_sentence_lengths)
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='avg. sentence length', ax=ax1)

        fig.suptitle('Avg. sentence length', fontsize=16)
        fig.tight_layout()
        plt.show()

    def plot_pronouns(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5))

        # Plot 1: First pronouns
        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == False][self.fake_news.content_label].apply(
            lambda x: sum([x.count(word) for word in self.stat.first_pronouns]))
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == True][self.fake_news.content_label].apply(
            lambda x: sum([x.count(word) for word in self.stat.first_pronouns]))
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='# of 1st pronouns', title='1st pronouns', ax=ax1)

        # Plot 2: Second pronouns
        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == False][self.fake_news.content_label].apply(
            lambda x: sum([x.count(word) for word in self.stat.second_pronouns]))
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == True][self.fake_news.content_label].apply(
            lambda x: sum([x.count(word) for word in self.stat.second_pronouns]))
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='# of 2nd pronouns', title='2nd pronouns', ax=ax2)

        # Plot 3: Third pronouns
        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == False][self.fake_news.content_label].apply(
            lambda x: sum([x.count(word) for word in self.stat.third_pronouns]))
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == True][self.fake_news.content_label].apply(
            lambda x: sum([x.count(word) for word in self.stat.third_pronouns]))
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='# of 3rd pronouns', title='3rd pronouns', ax=ax3)

        # Plot 4: Pronouns
        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == False][self.fake_news.content_label].apply(
            lambda x: sum([x.count(word) for word in self.stat.pronouns]))
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == True][self.fake_news.content_label].apply(
            lambda x: sum([x.count(word) for word in self.stat.pronouns]))
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='# of pronouns', title='total pronouns', ax=ax4)

        # Display the plots
        fig.suptitle('Personal pronouns', fontsize=16)
        fig.tight_layout()
        plt.show()

    def plot_negations(self):
        fig, ax1 = plt.subplots(1, 1, figsize=(3, 4))

        # Plot 1: First person pronouns
        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == False][self.fake_news.content_label].apply(
            lambda x: sum([x.count(word) for word in self.stat.negations]))
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == True][self.fake_news.content_label].apply(
            lambda x: sum([x.count(word) for word in self.stat.negations]))
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='# of negations', ax=ax1)

        fig.suptitle('Negations', fontsize=16)
        fig.tight_layout()
        plt.show()

    def plot_sentence_analysis(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))

        # Plot 1: Polarity score
        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == False][self.fake_news.sentence_analysis_label].apply(lambda x: x[0])
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == True][self.fake_news.sentence_analysis_label].apply(lambda x: x[0])
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='polarity score', title='Polarity score', ax=ax1)

        # Plot 2: Subjective score
        true = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == False][self.fake_news.sentence_analysis_label].apply(lambda x: x[1])
        fake = self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label]
                                   == True][self.fake_news.sentence_analysis_label].apply(lambda x: x[1])
        data = {"Reliable": true, "Fake": fake}
        self.stat.boxplot_data1_vs_data2(data=data, ylabel='subjective score', title='Subjective score', ax=ax2)

        fig.suptitle('Sentence analysis', fontsize=16)
        fig.tight_layout()
        plt.show()

    def plot_venn(self, nwords=40):
        fig, ax = plt.subplots(figsize=(8, 8))  # change figsize to a smaller size
        true_words, true_words_cnt = self.stat.sort_frequency(
            self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == False][self.fake_news.content_label].explode().tolist(), percentage=False)
        fake_words, fake_words_cnt = self.stat.sort_frequency(
            self.fake_news.data[self.fake_news.data[self.fake_news.binary_type_label] == True][self.fake_news.content_label].explode().tolist(), percentage=False)

        true_words = [word for word in true_words[:nwords]]
        fake_words = [word for word in fake_words[:nwords]]

        true_set = set(true_words)
        fake_set = set(fake_words)

        # Limit the set words to max to characters
        true_set = set([word for word in true_set])
        fake_set = set([word for word in fake_set])

        venn2([true_set, fake_set], set_labels=("Reliable", "Fake"))

        venn2_circles = venn2([true_set, fake_set], set_labels=("Reliable", "Fake"))

        # id='10' is the left circle, id='01' is the right circle, id='11' is the intersection
        venn2_circles.get_patch_by_id('10').set_color('red')
        venn2_circles.get_patch_by_id('10').set_alpha(0.3)
        venn2_circles.get_patch_by_id('01').set_color('blue')
        venn2_circles.get_patch_by_id('01').set_alpha(0.3)

        venn2_circles.get_patch_by_id('11').set_color('#9d0042')

        intersection_label = venn2_circles.get_label_by_id('11')
        intersection_label.set_text('\n'.join(list(true_set.intersection(fake_set))))
        intersection_label.set_fontsize(10)

        disjoint_words1 = set(true_words) - set(fake_words)
        disjoint_words2 = set(fake_words) - set(true_words)

        venn2_circles.get_label_by_id('01').set_text('\n'.join(list(disjoint_words1)))
        venn2_circles.get_label_by_id('10').set_text('\n'.join(list(disjoint_words2)))

        fig.tight_layout()
        fig.suptitle('Venn diagram of word frequency', fontsize=16)
        plt.show()


class Liar(Statistics):
    _initialized = False

    def __init__(self, data: pd.DataFrame, type_label: str = None, binary_type_label: str = None, statement_label: str = None,
                 subjects_label: str = None, speaker_label: str = None, party_label: str = None, sentence_analysis_label: str = None):
        self.data = data
        self.stat = Statistics()
        self.type_label = type_label
        self.binary_type_label = binary_type_label
        self.statement_label = statement_label
        self.subjects_label = subjects_label
        self.speaker_label = speaker_label
        self.party_label = party_label
        self.sentence_analysis_label = sentence_analysis_label
        # Convert text-list-of-strins to list of strings
        if not Liar._initialized:
            self.data[statement_label] = self.data[statement_label].apply(literal_eval)
            self.data[sentence_analysis_label] = self.data[sentence_analysis_label].apply(literal_eval)
            # Replace empty values with 'None':
            self.data[party_label] = self.data[party_label].fillna('none')
        types = [
            'pants-fire',
            'false',
            'mostly-false',
            'barely-true',
            'half-true',
            'mostly-true',
            'true',
        ]
        # Assign colors to each type from the Tab20 color map
        colors = plt.get_cmap('tab10').colors[:len(types)]
        # Add black color for unknown types
        unknown_types = set(types)  # get all other unknown types
        colors += tuple(['black'] * len(unknown_types))  # add black color for other unknown types
        # Create a dictionary mapping each type to its assigned color
        self.types_colors = {types[i]: colors[i] for i in range(len(types))}

    def barplot_type(self, percentage: bool = True):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        types = self.data[self.type_label].explode().tolist()
        types, measure = self.stat.sort_frequency(text=types, percentage=percentage)
        color_list = [self.types_colors[tp] for tp in types]
        ax.barh(types, measure, color=color_list, alpha=0.5)
        ax.set_xlabel('% of total labels' if percentage else '# of labels')
        ax.set_title('Label distribution', fontsize=16)
        fig.tight_layout()
        plt.show()


class Statistics_FakeNewsCorpus_vs_Liar(Statistics):
    def __init__(self, fake_news: FakeNewsCorpus, liar: Liar):
        self.stat = Statistics()
        self.fake_news = fake_news
        self.liar = liar

    def barplot_word_frequency(self, nwords=25):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
        words_fake_news, words_cnt_fake_news = self.stat.sort_frequency(
            self.fake_news.data[self.fake_news.content_label].explode(), percentage=True)
        words_liar, words_cnt_liar = self.stat.sort_frequency(
            self.liar.data[self.liar.statement_label].explode(), percentage=True)
        max_value = max(max(words_cnt_fake_news), max(words_cnt_liar))*1.1
        self.stat.barplot(data=words_fake_news, measure=words_cnt_fake_news, num=nwords, minmax=(
            0, max_value), label='% of total words', title='FakeNewsCorpus', color='yellowgreen', ax=ax1)
        self.stat.barplot(data=words_liar, measure=words_cnt_liar, num=nwords, minmax=(
            0, max_value), label='% of total words', title='LIAR', color='orange', ax=ax2)

        fig.suptitle('Word frequency', fontsize=16)
        fig.tight_layout()
        plt.show()

    def boxplot_word_frequency(self):
        fig, ax1 = plt.subplots(1, 1, figsize=(3, 5))
        fake_news = self.fake_news.data[self.fake_news.content_label].apply(len)
        liar = self.liar.data[self.liar.statement_label].apply(len)
        data = {"FakeNewsCorpus": fake_news, "LIAR": liar}
        self.stat.boxplot_data1_vs_data2(data=data, colors=['yellowgreen', 'orange'], ylabel='# of words', ax=ax1)

        fig.suptitle('# of words', fontsize=16)
        fig.tight_layout()
        plt.show()

    def plot_party(self, nparties=25):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
        true_party, true_party_cnt = self.stat.sort_frequency(
            self.liar.data[self.liar.data[self.liar.binary_type_label] == False][self.liar.party_label].fillna('none').explode(), percentage=True)
        fake_party, fake_party_cnt = self.stat.sort_frequency(
            self.liar.data[self.liar.data[self.liar.binary_type_label] == True][self.liar.party_label].fillna('none').explode(), percentage=True)
        max_val = max(true_party_cnt + fake_party_cnt)*1.1
        self.stat.barplot(data=true_party, measure=true_party_cnt, num=nparties, minmax=(0, max_val),
                          label='% of reliable articles', title='Reliable', color='lightsalmon', ax=ax1)
        self.stat.barplot(data=fake_party, measure=fake_party_cnt, num=nparties, minmax=(
            0, max_val), label='% of fake articles', title='Fake', color='lightblue', ax=ax2)

        fig.suptitle('Party frequency', fontsize=16)
        fig.tight_layout()
        plt.show()

    def plot_speaker(self, nspeakers=25):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
        true_speaker, true_speaker_cnt = self.stat.sort_frequency(
            self.liar.data[self.liar.data[self.liar.binary_type_label] == False][self.liar.speaker_label].explode(), percentage=True)
        fake_speaker, fake_speaker_cnt = self.stat.sort_frequency(
            self.liar.data[self.liar.data[self.liar.binary_type_label] == True][self.liar.speaker_label].explode(), percentage=True)
        max_val = max(true_speaker_cnt + fake_speaker_cnt)*1.1
        self.stat.barplot(data=true_speaker, measure=true_speaker_cnt, num=nspeakers, minmax=(
            0, max_val), label='% of reliable articles', title='Reliable', color='lightsalmon', ax=ax1)
        self.stat.barplot(data=fake_speaker, measure=fake_speaker_cnt, num=nspeakers, minmax=(
            0, max_val), label='% of fake articles', title='Fake', color='lightblue', ax=ax2)

        fig.suptitle('Speaker frequency', fontsize=16)
        fig.tight_layout()
        plt.show()

    def plot_subjects(self, nsubjects=25):
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))
        subjects, subjects_cnt = self.stat.sort_frequency(
            self.liar.data[self.liar.subjects_label].explode(), percentage=True)
        self.stat.barplot(data=subjects, measure=subjects_cnt, num=nsubjects,
                          label='% of total subjects', color='lightsalmon', ax=ax1)

        fig.suptitle('Subjects frequency', fontsize=16)
        fig.tight_layout()
        plt.show()

    def plot_type(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3))
        fake_news_words, fake_news_words_cnt = self.stat.sort_frequency(
            self.fake_news.data[self.fake_news.binary_type_label].tolist(), percentage=True)
        self.stat.barplot(data=fake_news_words, measure=fake_news_words_cnt, boolean=True,
                          title="FakeNewsCorpus", label="% of total labels", ax=ax1)
        liar_words, liar_words_cnt = self.stat.sort_frequency(
            self.liar.data[self.liar.binary_type_label].tolist(), percentage=True)
        self.stat.barplot(data=liar_words, measure=liar_words_cnt, boolean=True,
                          title="LIAR", label="% of total labels", ax=ax2)

        fig.suptitle('Binary Label distribution', fontsize=16)
        fig.tight_layout()
        plt.show()
