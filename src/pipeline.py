#import filehandling as fh
import preprocessing as pp
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from ast import literal_eval
import numpy as np
from sklearn.preprocessing import normalize

tqdm.pandas()

headers = {
    'row': 0,
    'id': 1,
    'domain': 2,
    'type': 3,
    'url': 4,
    'content': 5,
    'scraped_at': 6,
    'inserted_at': 7,
    'updated_at': 8,
    'title': 9,
    'authors': 10,
    'keywords': 11,
    'meta_keywords': 12,
    'tags': 13,
    'summary': 14
}


class FunctionApplier:
    def function_to_apply(self, row):
        pass


class Normalize(FunctionApplier):
    def function_to_apply(self, vector):
        # Compute the sum of all elements in the input vector
        sum = np.sum(vector)
        # Check if the sum is zero to avoid division by zero
        if sum == 0:
            return vector
        # Normalize the input vector by dividing each element by the sum
        return vector / sum


class Create_word_vector(FunctionApplier):
    def __init__(self, unique_words):
        self.unique_words = unique_words

    def function_to_apply(self, words):
        vector = np.zeros(len(self.unique_words), dtype=int)
        words = sorted(words)
        i = 0
        j = 0
        while i < len(words) and j < len(self.unique_words):
            if words[i] == self.unique_words[j]:
                vector[j] += 1
                i += 1
            elif words[i] > self.unique_words[j]:
                j += 1
            else:  # should never happen
                i += 1
        return np.array(vector)

class Generate_unique_word_list(FunctionApplier):
    def __init__(self):
        self.unique_words = Counter()

    def function_to_apply(self, words):
        self.unique_words.update(words)
        return words

    def get_unique_words(self, low, high):
        # Get the sum of all words
        word_sum = sum(self.unique_words.values())
        # Sort the words by frequency and filter out the words that are not within the given range
        sorted_items = sorted(self.unique_words.items(),
                              key=lambda x: x[1], reverse=True)
        sorted_freq_items = [x[0] for x in sorted_items if x[1] /
                             word_sum >= low and x[1] / word_sum <= high]

        return sorted(sorted_freq_items)

    def get_freqs(self):
        # Get the sum of all words
        word_sum = sum(self.unique_words.values())
        # Sort the words by frequency and filter out the words that are not within the given range
        sorted_items = sorted(self.unique_words.items(),
                              key=lambda x: x[1], reverse=True)
        return [(x[0], x[1] / word_sum) for x in sorted_items]

    def get_most_frequent(self, nwords):
        # Return the n most frequent words
        return sorted(self.unique_words.most_common(nwords))

    def plot_most_frequent(self, nwords, freq=False):
        # Calculate the frequency of each word
        words = [x[0] for x in self.unique_words.most_common(nwords)]
        # Calculate the frequency of each word
        frequency = [x[1] for x in self.unique_words.most_common(nwords)]
        # If freq is True, normalize the frequency
        if freq:
            s = sum(self.unique_words.values())
            frequency = [x / s for x in frequency]
        # Plot the frequency of the n most frequent words
        plt.bar(words, frequency)
        plt.ylabel('Frequency')
        plt.title(f'Frequency of the {nwords} most frequent words')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_frequency_line(self, nwords):
        # Calculate the frequency of each word
        word_sum = sum(self.unique_words.values())
        frequency = [
            x[1] / word_sum for x in self.unique_words.most_common(nwords)]
        # Plot the frequency of the n most frequent words
        plt.plot(list(range(len(frequency))), frequency)
        plt.ylabel('Frequency')
        plt.title(f'Frequency of the {nwords} most frequent words')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


class Word_frequency(FunctionApplier):
    def __init__(self, nwords=50):
        self.swords = nwords
        self.words = []
        self.frequency = Counter()
        self.sorted_frequency = []

    def function_to_apply(self, content):
        # Update/add list of word
        content = literal_eval(content)
        self.frequency.update(content)
        # Return the sorted dictionary based on the frequency of each word
        self.sorted_frequency = sorted(
            self.frequency.items(), key=lambda x: x[1], reverse=True)
        return content

    def plot(self):
        # Extract the words and their frequency from the sorted list
        words = [x[0] for x in self.sorted_frequency[:self.swords]]
        frequency = [x[1] for x in self.sorted_frequency[:self.swords]]
        # Plot a barplot using matplotlib
        plt.bar(words, frequency)
        plt.ylabel('Frequency')
        plt.title(f'Frequency of the {self.swords} most frequent words')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


class Tokenizer(FunctionApplier):
    def function_to_apply(self, cell):
        return cell.split()


class Remove_stopwords(FunctionApplier):
    def __init__(self, swords):
        self.swords = swords

    def function_to_apply(self, words):
        return [w for w in words if not w in self.swords]


class Stem(FunctionApplier):
    def function_to_apply(self, words: list[str]):
        # Create a PorterStemmer object, which remove morphological affixes from words, leaving only the word stem.
        ps = PorterStemmer()
        stemmed_words = []
        for w in words:
            stemmed_words.append(ps.stem(w))
        return stemmed_words


class Clean_data(FunctionApplier):
    def __init__(self):
        # Create a list of patterns to remove.
        # Compile the patterns to speed up the process
        self.patterns = {
            re.compile(r'((https?:\/\/)?(?:www\.)?[a-zA-Z0-9-_\+=.:~@#%]+\.[a-zA-Z0-9()]{1,6}\b(?:[a-zA-Z0-9-_.:\\/@#$%&()=+~?]*))'): ' <URL> ',
            re.compile(r'(https?:\/\/)?w{0,3}\.?[a-z]+\.[a-z]\w*[\w\/-]*'): ' <URL> ',
            re.compile(r'(\d{1,2}([\:\-/\\]|(,\s)?)){2}\d{2,4}|\d{2,4}(([\:\-/\\]|(,\s)?)\d{1,2}){2}'): ' <DATE> ',
            re.compile(r'([Jj]an(uary)?|[Ff]eb(ruary)?|[Mm]ar(ch)?|[Aa]pr(il)?|[Mm]ay|[Jj]un(e)?|[Jj]ul(y)?|[Aa]ug(ust)?|[Ss]ep(tember)?|[Oo]ct(ober)?|[Nn]ov(ember)?|[Dd]ec(ember)?)([\:\-/\\]|(,\s)?)\d{1,2}([\:\-/\\]|(,\s)?)\d{1,4}'): ' <DATE> ',
            re.compile(r'([\w.\-]+@(?:[\w-]+\.)+[\w-]{2,4})|@[\w\d]+'): ' <EMAIL> ',
            re.compile(r'(\r\n|\n|\r)+'): ' ',
            re.compile(r'(\t+)'): ' ',
            re.compile(r'(\!|\[|\])'): '',
            # re.compile(r'(\=|\~|\u2018|\t|\;|\@|\″|\^|\…|\<|\>|\+|\/|\.|\*|\#|\,|\?|\&|\"|\”|\“|\%|\:|\-|\(|\)|\´|\`|\’|\$|\'|\|)'): '',
            # re.compile(r'(\–|\—)'): ' ',
            re.compile(r'[^A-Za-z0-9\s]'): '',
            re.compile(r'(\d+)(th)?'): ' <NUM> ',
            re.compile(r'( +)'): ' ',
        }

    def function_to_apply(self, cell):
        # Apply patterns using list comprehension
        cell = str(cell)
        cell = cell.lower()
        # Loop through each pattern and apply the pattern to each row and do replacement if needed
        for pattern, replacement in self.patterns.items():
            cell = re.sub(pattern, replacement, cell)

        return cell


class Decode_to_str(FunctionApplier):
    def function_to_apply(self, row):
        return row.decode("utf-8")


class Decode_from_json(FunctionApplier):
    def function_to_apply(self, row):
        return json.loads(row)


class Encode_to_json(FunctionApplier):
    def function_to_apply(self, row):
        return json.dumps(row)


class Print_first_row(FunctionApplier):
    def __init__(self):
        self.has_printed = False

    def function_to_apply(self, row):
        if not self.has_printed:
            self.has_printed = True


class Print_content_to_csv(FunctionApplier):
    def __init__(self, num_to_print, csv_file):
        self.has_printed = False
        self.table = []
        self.data_frame = pd.DataFrame()
        self.csv_file = csv_file
        self.num_to_print = num_to_print

    def function_to_apply(self, row):
        if self.num_to_print > 0:
            self.num_to_print -= 1
            item = {}
            for h, i in headers.items():
                item[h] = row[i]
            self.table.append(item)

        elif not self.has_printed:
            self.has_printed = True
            self.data_frame = pd.DataFrame(data=self.table)
            self.data_frame.to_csv(self.csv_file)

        return row


class Binary_labels(FunctionApplier):
    def __init__(self):
        self.binary_labels: dict = {
            'fake': False,
            'conspiracy': False,
            'junksci': False,
            'hate': False,
            'unreliable': False,
            'bias': False,
            'satire': False,
            'state': False,
            'reliable': True,
            'clickbait': True,
            'political': True
        }

    def function_to_apply(self, cell):
        try:
            binary_label = self.binary_labels[cell]
        except:
            # TODO: what to do when no labels
            #print("Key error in binary_labels class:", cell)
            binary_label = True
        return binary_label

class Simple_model(FunctionApplier):
    def __init__(self):
        self.dict_domains = {}

    def function_to_apply(self, row):
        # TODO ASKE DO YOUR THING

        return row

    def get_metrics(self):
        pass


ROWS_PR_ITERATION = 20000
ROWS = 8529853
TQDM_COLOR = 'magenta'


def get_batch(df, batch_size):
    new_df = pd.DataFrame()
    grouped = df.groupby('type', axis=0)
    for name, group in grouped:
        new_group = group.sample(n=min(batch_size, len(group)))
        new_df = pd.concat([new_df, new_group], ignore_index=True)
    return new_df


def get_batch_from_csv(file: str, batch_size: int):
    return get_batch(pd.read_csv(file), batch_size)


def applier(function_cols, chunk, progress_bar=False):
    # Apply the specified functions to each column or row in the chunk
    for f in function_cols:
        if len(f) == 2:
            function, col = f
            if col is None:
                if progress_bar:
                    chunk = chunk.progress_apply(function.function_to_apply, axis=1)
                else:
                    chunk = chunk.apply(function.function_to_apply, axis=1)
            else:
                if progress_bar:
                    chunk[col] = chunk[col].progress_apply(function.function_to_apply)
                else:
                    chunk[col] = chunk[col].apply(function.function_to_apply)
        elif len(f) == 3:
            function, from_col, to_col = f
            if progress_bar:
                chunk[to_col] = chunk[from_col].progress_apply(function.function_to_apply)      
            else:
                chunk[to_col] = chunk[from_col].apply(function.function_to_apply)


def applier(function_cols, chunk):
    # Apply the specified functions to each column or row in the chunk
    for function, col in function_cols:
        if col is None:
            chunk = chunk.apply(function.function_to_apply, axis=1)
        else:
            chunk[col] = chunk[col].apply(function.function_to_apply)
    return chunk


def progress_applier(function_cols, chunk):
    # Apply the specified functions to each column or row in the chunk
    for function, col in function_cols:
        if col is None:
            chunk = chunk.progress_apply(function.function_to_apply, axis=1)
        else:
            chunk[col] = chunk[col].progress_apply(function.function_to_apply)
    return chunk


def apply_pipeline_pd(df, function_cols):
    # Iterate through each row in the DataFrame and apply the functions
    return applier(function_cols, df.copy())


def apply_pipeline_pd_tqdm(df, function_cols):
    # Iterate through each row in the DataFrame and apply the functions
    return applier(function_cols, df.copy(), progress_bar=True)

def apply_pipeline(old_file, function_cols, new_file=None, batch_size=ROWS_PR_ITERATION, get_batch=False):
    i = 0
    start_time = time()

    # Use Pandas chunksize and iterator to read the input file in batches
    with pd.read_csv(old_file, chunksize=batch_size, encoding='utf-8', lineterminator='\n') as reader:
        for chunk in reader:
            if function_cols is None:
                return chunk
            # Apply the specified functions to each row in the batch
            chunk = applier(function_cols, chunk)
            # If an output file is specified, append the processed data to it
            if new_file is not None:
                if i == 0:
                    chunk.to_csv(new_file, mode='w', index=False)
                else:
                    # Append to csv file without header
                    chunk.to_csv(new_file, mode='a', header=False, index=False)
            # If get_batch is True, return only the first batch of processed data
            if get_batch:
                print("Length", len(chunk))
                return chunk

            i += batch_size
        # Print the time taken to process the data
        print(f'finish time: {time()-start_time}')


def read_rows_of_csv(file, n=None):
    return pd.read_csv(file, nrows=n) if n is not None else pd.read_csv(file)


def create_csv_from_existing_with_n_rows(file, new_file, n):
    df = read_rows_of_csv(file, n)
    df.to_csv(new_file)


def create_test_file():
    create_csv_from_existing_with_n_rows(   
        "../datasets/big/news_cleaned_2018_02_13.csv", "../datasets/big/news_sample.csv", 100)
    print(read_rows_of_csv("../datasets/big/news_sample.csv")["content"])


def ist_pipeline():
    stopwords_lst = stopwords.words('english') + ["<NUM>", "<DATE>", "<URL>"]
    apply_pipeline("../datasets/big/news_sample.csv", [
        (Clean_data(), "content"),
        (Tokenizer(), "content"),
        (Remove_stopwords(stopwords_lst), "content"),
        (Stem(), "content"),
    ], new_file="../datasets/sample/news_sample_cleaned_num.csv")


def word_freq_pipeline():
    wf = Word_frequency()
    apply_pipeline("../datasets/big/news_sample_cleaned.csv", [
        (wf, "content")
    ],
    new_file="../datasets/1mio-raw-cleaned-freq.csv"
    )
    wf.plot()


def simple_model_test():
    sm = Simple_model()
    apply_pipeline("../datasets/big/news_sample_cleaned.csv", [
        (sm, None)
    ], )
    sm.get_metrics()


#unique_words = Generate_unique_word_list()
