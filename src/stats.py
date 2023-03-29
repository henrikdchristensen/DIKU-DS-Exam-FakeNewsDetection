import preprocessing as pp
import pipeline as pl
from pipeline import FunctionApplier, apply_pipeline, ist_pipeline
import matplotlib.pyplot as plt
from collections import Counter
from ast import literal_eval
import numpy as np
from scipy.stats import norm
from matplotlib_venn import venn2
import pandas as pd
from multiprocessing import Process
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
# import StandardScaler
from sklearn.preprocessing import StandardScaler

# import count vectorizer
from sklearn.feature_extraction.text import CountVectorizer



RAW_DATA = '../datasets/sample/dataset.csv'
CLEANED_DATA = '../datasets/sample/news_sample_cleaned.csv'
CLEANED_DATA_NUM = '../datasets/sample/news_sample_cleaned_num_100k.csv'


class Word_frequency(FunctionApplier):
    def __init__(self, nwords = 50):
        self.swords = nwords
        self.words = []
        self.frequency = Counter()
        self.sorted_frequency = []
        self.totalRows = 0

    def function_to_apply(self, content):
        # Update/add list of word
        # print(type(content))
        content: list = literal_eval( content )
        # content = [x for x in content if x != "<number>"]
        # print(content)
        self.frequency.update(content)
        # Return the sorted dictionary based on the frequency of each word
        self.sorted_frequency = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        # print("sorted_frequency", self.sorted_frequency)
        self.totalRows += 1
        return content

    def function_to_apply_df(self, content):
        # Update/add list of word
        # content: list = list[content]
        # content = [x for x in content if x != "<number>"]
        print(content)
        content = content
        print(content)
        self.frequency.update(content)
        # Return the sorted dictionary based on the frequency of each word
        self.sorted_frequency = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        # print("sorted_frequency", self.sorted_frequency)
        self.totalRows += 1
        return content
    
    def plot(self):
        # Extract the words and their frequency from the sorted list
        # print(self.sorted_frequency)
        words = [x[0] for x in self.sorted_frequency[:self.swords]]
        frequency = [x[1] for x in self.sorted_frequency[:self.swords]]
        # Plot a barplot using matplotlib
        plt.bar(words, frequency)
        plt.ylabel('Frequency')
        plt.title(f'Frequency of the {self.swords} most frequent words')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plotVenn(self, other_sorted_frequency): 
        words = [x[0] for x in self.sorted_frequency[:self.swords]]
        frequency = [x[1] for x in self.sorted_frequency[:self.swords]]

        other_words = [x[0] for x in other_sorted_frequency[:self.swords]]
        other_frequency = [x[1] for x in other_sorted_frequency[:self.swords]]

        set1 = set(words)
        set2 = set(other_words)

        venn2([set1, set2], set_labels=("Reliable", "Fake"))

        venn2_circles = venn2([set1, set2], set_labels=("Reliable", "Fake"))
        venn2_circles.get_patch_by_id('10').set_color('orange')
        venn2_circles.get_patch_by_id('10').set_alpha(0.5)
        
        intersection_label = venn2_circles.get_label_by_id('11')
        intersection_label.set_text('\n'.join(list(set1.intersection(set2))))
        
        # intersection_label.set_position((0.08, 0))
        disjoint_words1 = set(words) - set(other_words)
        disjoint_words2 = set(other_words) - set(words)

        # set the text of the disjoint words
        venn2_circles.get_label_by_id('01').set_text('\n'.join(list(disjoint_words1)))
        
        venn2_circles.get_label_by_id('10').set_text('\n'.join(list(disjoint_words2)))
        # set the text of the intersection int the middle of the venn diagram 
        intersection_label.set_fontsize(10)
        # venn2_circles.get_label_by_id('10').set_fontsize(14)

        plt.show()

    def plotWordMap():

        # Define a dictionary of word frequencies
        word_freq = {'word1': 10, 'word2': 5, 'word3': 7, 'word4': 3, 'word5': 12}

        # Create a WordCloud object
        wordcloud = WordCloud(background_color='white')

        # Generate a word map from the word frequency dictionary
        wordcloud.generate_from_frequencies(word_freq)

        # Display the word map
        import matplotlib.pyplot as plt
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    # plot the frequency of the words from the fake news and plot the frequency of the words from the real news
    def plot_fake_real(self, other_sorted_frequency, set_labels = ("Fake", "Reliable") ):
        # Extract the words and their frequency from the sorted list
        words = [x[0] for x in self.sorted_frequency[:self.swords]]
        frequency = [x[1] for x in self.sorted_frequency[:self.swords]]

        other_words = [x[0] for x in other_sorted_frequency[:self.swords]]
        other_frequency = [x[1] for x in other_sorted_frequency[:self.swords]]

        disjoint_words = set(words) - set(other_words)
        print("disjoint_words", disjoint_words)

        # map the word frequency from the fake news word list to the words from the real news
        for i in range(len(words)):
            if not words[i] in other_words:
                other_frequency[i] = 0

        # Set the width of the bars
        bar_width = 0.35

        print("freq", frequency)
        print("\n\n")
        print("otherfreq", other_frequency)

        # Set the positions of the bars on the x-axis
        fake_pos = np.arange(len(words))
        reliable_pos = fake_pos + bar_width

        # Create the figure and axis objects
        fig, ax = plt.subplots()

        # Plot the bars for fake news
        ax.bar(fake_pos, frequency, width=bar_width, color='b', label=set_labels[0])

        # Plot the bars for reliable news
        ax.bar(reliable_pos, other_frequency, width=bar_width, color='g', label=set_labels[1])

        # Add labels and title to the plot
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_xticks(fake_pos + bar_width / 2)
        ax.set_xticklabels(words)
        ax.set_title('Word Frequency Comparison')
        
        # rotate the xticks
        plt.xticks(rotation=90)

        # Add a legend to the plot
        ax.legend()

        # Show the plot
        plt.show()

class Count_Items(FunctionApplier):
    def __init__(self):
        self.count = {
            "urls": 0,
            "dates": 0,
            "numbers": 0 
        }

    def function_to_apply(self, content):
        # Update/add list of word
        content: list = literal_eval(str(content))
        self.countItems(content)
        

    def countItems(self, content):
        # pp.Exploration.countItems(data)
        for text in content:
            self.count["urls"] += text.count("<url>")
            self.count["dates"] += text.count("<date>")
            self.count["numbers"] += text.count("<num>")


class Contribution(FunctionApplier): 
    def __init__(self):
        self.data = []

    def function_to_apply(self, content: pd.DataFrame):
        self.data.append(content.copy())

    def contributionPlot2(self):
        
        threshold = 2
        self.data = pd.DataFrame(self.data)

        # print("got keys: \n", len(self.data), self.data.iloc[0].keys())

        # group the articles by domain and category, and count the number of articles in each group
        # print("groupby: \n", self.data['domain'].unique())
        counts = self.data.groupby(['domain', 'type'])['content'].count().unstack()

        # print("counts: \n", counts)

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

    def contributionPlot1(self):

        threshold = 3
        self.data = pd.DataFrame(self.data)

        # group the articles by type and domain, and count the number of articles in each group
        type_groups = self.data.groupby(['type', 'domain'])['content'].count()
        type_groups = type_groups.unstack().fillna(0)
            
        # iterate over each unique type to generate a new DataFrame with the count of each domain for that type
        domain_counts = []
        for typ in self.data['type'].unique():
            type_counts = type_groups.loc[typ].rename(typ)
            domain_counts.append(type_counts)

        # combine the domain counts into a new DataFrame where the rows are each unique domain and the columns are the counts for each type
        domain_counts = pd.concat(domain_counts, axis=1, sort=False)

        # sum the counts for each domain across all types and sort the resulting Series in descending order
        domain_totals = domain_counts.sum(axis=1)
        domain_totals = domain_totals.sort_values(ascending=False)

        # print domain_totals and domain_counts and compare the order of the domains
        print(domain_totals)
        print("\n\n")
        print(domain_counts)
        print("\n\n")

        print("total index: ", domain_totals.index)

        # reorder the columns of the DataFrame using the sorted Series
        # domain_counts = domain_counts[domain_counts.index]

        percentages = domain_counts.apply(lambda x: x / x.sum() * 100).round(2)

        # filter the percentages to only show the contributions above the threshold
        percentages = percentages[percentages > threshold]
        # drop the rows with all NaN values
        percentages = percentages.dropna(how='all')


        # for 
        # [reliable  political  conspiracy   fake   bias  unreliable  junksci  clickbait   hate  satire]

        for type in self.data['type'].unique():
            # print(type)
            # print(domain_counts[type].sort_values(ascending=False))
            percentages.sort_values(type, na_position='first', ascending=False, inplace=True)
        
        print("percentages: ", percentages)
        # create a stacked horizontal bar chart of the percentages
        ax = percentages.plot(kind='barh', stacked=True, figsize=(10, 8))
        # set the x-axis label to show the percentages
        ax.set_xlabel('Percentage')
        # set the legend to display outside the chart
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        
        title = f'Contribution of Domains to Categories ( ≥ {threshold}%)'
        ax.set_title(title)

        # show the chart
        plt.show()


# make class to count each type fake, real, unreliable, reliable etc. and make a frequency plot
class Article_Type_frequency(FunctionApplier):
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

class detectOutliers(FunctionApplier):
    def __init__(self):
        self.outliers = []
        self.outliers_index = []
        self.outliers_value = []

    def function_to_apply(self, value):
        # print(value)
        if value > 0.5:
            self.outliers.append(value)
            self.outliers_index.append(self.index)
            self.outliers_value.append(self.value)
        return value
    
    def getDataFrame(self, csv):
        df = pd.read_csv(csv, nrows=20000)
        return df

    
    def kMeans(self):

        df = self.getDataFrame("cleaned_fake_news.csv")

        
        # Load cleaned fake news corpus
        # df = pd.read_csv("cleaned_fake_news.csv")

        # Vectorize the corpus using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(df['text'])

        # Perform K-Means clustering
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
        kmeans.fit(tfidf)

        # Print the top terms in each cluster
        print("Top terms per cluster:")
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = tfidf_vectorizer.get_feature_names()
        for i in range(num_clusters):
            print(f"Cluster {i} terms: ", end='')
            for j in order_centroids[i, :10]:
                print(f"{terms[j]}, ", end='')
            print()

        # Visualize the clusters
        plt.scatter(tfidf[:, 0], tfidf[:, 1], c=kmeans.labels_, cmap='rainbow')
        plt.show()

        # X = fake_news_df[['type', 'title', 'text']]
        # X_scaled = StandardScaler().fit_transform(X)
        # # Initialize the k-means model with the desired number of clusters
        # kmeans = KMeans(n_clusters=3)

        # # Fit the model to the data
        # kmeans.fit(X_scaled)

        # # Identify the cluster labels for each data point
        # labels = kmeans.labels_

        # # Add the cluster labels to the original dataframe
        # fake_news_df['cluster_label'] = labels


def clean():
    ist_pipeline(RAW_DATA)

def plot_in_processes(plot_funcs):
    """Spawn multiple processes to plot matplotlib plots."""
    processes = []
    for plot_func in plot_funcs:
        process = Process(target=plot_func)
        process.start()
        processes.append(process)
    for process in processes:
        process.join()




def wordFrequency():

    size = 100000
    blc = pl.Binary_labels()
    bl = blc.binary_labels


    # get all values that are true in the binary labels
    trueFilterList = [k for k, v in bl.items() if v]
    print(trueFilterList)

    # get all values that are false in the binary labels
    falseFilterList = [k for k, v in bl.items() if not v]
    print(falseFilterList)

    trueFilter = pl.Filter(trueFilterList)
    falseFilter = pl.Filter(falseFilterList)
    
    wf = Word_frequency()
    apply_pipeline(
        CLEANED_DATA_NUM, 
        [
            (falseFilter, "type"),
            (wf, "content"),
        ],
        batch_size=size,
        # get_batch=True,
        # type="fake"
        # exclude=["reliable"]
        # total_rows=20000,
        total_rows=20000,
        progress_bar=True
    )

    wf2 = Word_frequency()
    apply_pipeline(
        CLEANED_DATA_NUM, 
        [
            (trueFilter, "type"),
            (wf2, "content"),
        ],
        # batch_size=size,
        total_rows=20000,
        progress_bar=True
        # get_batch=True,
        # type="reliable"
    )

    return wf, wf2
    # cp = Contribution()
    # apply_pipeline(
    #     CLEANED_DATA_NUM, 
    #     [
    #         (cp, None),
    #     ],
    #     batch_size=10000,
    #     progress_bar=True,
    #     # get_batch=True,
    #     # type="reliable"
    # )


    # print(ss.count)

    # atf.plotDistribution()
    # atf.plotDistribution()
    # wf.plot()

    # wf.plotVenn(wf2.sorted_frequency)

    # print(cp.data.iloc[0:10])

    # cp.contributionPlot2()
    # wf2.plot_fake_real(wf.sorted_frequency, set_labels=("Reliable", "Fake"))
    # wf.plot()

    # wf.plot_fake_real(wf2.sorted_frequency)
    # plot_in_processes([cp.contributionPlot2, wf.plot])



def runStats2():
    
    wf = Word_frequency()
    ss = Count_Items()
    atf = Article_Type_frequency()
    apply_pipeline(
        CLEANED_DATA, 
        [
            (wf, "content"),
            (ss, "content"),
            (atf, "type")
        ],
        batch_size=10000,
        # get_batch=True
    )

    print(ss.count)

    # atf.plotDistribution()
    # atf.plotDistribution()
    wf.plot()


# runStats()
# clean()



