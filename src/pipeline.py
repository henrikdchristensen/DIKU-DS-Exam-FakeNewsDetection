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
import bisect
import nltk
nltk.download('punkt')

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

labels: dict = {
    'fake': False,
    'conspiracy': False,
    'junksci': False,
    'hate': False,
    'unreliable': False,
    'bias': False,
    'satire': False,
    #'state': False,
    'reliable': True,
    'clickbait': True,
    'political': True
}

ROWS_PR_ITERATION = 200000
ROWS = 8529853
TQDM_COLOR = 'magenta'
DELETE_TOKEN = '<DELETE>'

class FunctionApplier:
    def function_to_apply(self, row):
        pass

class Filter(FunctionApplier):
    def __init__(self, include):
        self.include = include

    def function_to_apply(self, type):
        if type not in self.include:
            return DELETE_TOKEN        
        return type


class Normalize(FunctionApplier):
    def function_to_apply(self, vector):
        # Compute the sum of all elements in the input vector
        sum = np.sum(vector)
        # Check if the sum is zero to avoid division by zero
        if sum == 0:
            return vector
        # Normalize the input vector by dividing each element by the sum
        return vector / sum

class TF_IDF(FunctionApplier):
    def __init__(self, n, n_t, unique_words):
        self.unique_words = unique_words
        self.idf_vec = self.get_idf(n, n_t)

    def get_idf(self, n, n_t):        
        idf_vec = [np.zeros(len(self.unique_words), dtype=int) for _ in range(len(n))]
        for i in range(len(n)):
            for j, word in enumerate(self.unique_words):
                if word in n_t[i]:
                    idf_vec[i][j] = np.log(n[i] / n_t[i][word])
                else:
                    idf_vec[i][j] = 0
        return idf_vec

    def function_to_apply(self, row):
        vector = row["content"].copy()
        set = row["set"]
        for i in range(len(vector)):
            if vector[i] > 0:
                vector[i] = (np.log(vector[i]) + 1) * self.idf_vec[set][i]
        return vector

def get_dataframe_with_distribution(file, total_size, splits, balanced, end_col = "set", type_col ="type",  chunksize=ROWS_PR_ITERATION, out_file=None, get_frame=True, classes = labels, delete=True):
    # empty dataframe
    data = None
    curr_index = 0
    sets = []
    for split, b in zip(splits, balanced):
        if not b:
            sets.append([b, int(split * total_size)])
        else:
            split_dict = {}
            label_num = int((split * total_size) / len(classes))
            for label in classes:
                split_dict[label] = label_num
            sets.append([b, split_dict])
    print(sets)
    def apply_to_rows(label):
        nonlocal curr_index
        if curr_index >= len(sets) or label not in classes:
            return DELETE_TOKEN
        
        balanced, curr_set = sets[curr_index]
        if balanced:
            if sum(curr_set.values()) == 0:
                curr_index += 1
                return apply_to_rows(label)
            elif curr_set[label] > 0:
                curr_set[label] -= 1
                return curr_index
            else:
                return DELETE_TOKEN
        else:
            if curr_set == 0:
                curr_index += 1
                return apply_to_rows(label)
            elif curr_set > 0:
                sets[curr_index][1] -= 1 
                return curr_index
            else:
                return DELETE_TOKEN
    
    entries_read = 0
    with pd.read_csv(file, chunksize=chunksize, encoding='utf-8') as reader:
        for chunk in reader:
            chunk[end_col] = chunk[type_col].progress_apply(apply_to_rows)
            if delete:
                chunk = chunk[chunk[end_col] != DELETE_TOKEN]

            if out_file is not None:
                if entries_read == 0:
                    chunk.to_csv(out_file, mode='w', index=False)
                else:
                    # Append to csv file without header
                    chunk.to_csv(out_file, mode='a', header=False, index=False)
            if get_frame:
                if data is None:
                    data = chunk
                else:
                    data = pd.concat([data, chunk])

            entries_read += chunksize
            finished = True
            for balanced, set in sets:
                if (balanced and sum(set.values()) > 0) or (not balanced and set > 0):
                    finished = False
            if finished:
                print("entries read:", entries_read)
                print(sets)
                if get_frame:
                    return data
                return
    print("ERROR: not enough data to create sets")
    return data

class Debug(FunctionApplier):
    def __init__(self):
        self.i=0
    def function_to_apply(self, row):
        if type(row) != str:
            print(self.i, row)
        self.i += 1
        return row

class Read_String_Lst(FunctionApplier):
    def function_to_apply(self, words):
        if type(words) is not list:
            words = literal_eval(words)
        return words
    
class Combine_Content(FunctionApplier):
    def function_to_apply(self, content_lst):
        if content_lst == []: # to avoid nan
            return " "
        return " ".join(content_lst)

class Create_word_vector(FunctionApplier):
    def __init__(self, unique_words):
        self.unique_words = unique_words

    def function_to_apply(self, words):
        if type(words) is not list:
            words = literal_eval(words)
        vector = np.zeros(len(self.unique_words), dtype=np.int32)
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
        return vector


class Save_numpy_arr(FunctionApplier):
    def function_to_apply(self, vector):
        return ' '.join(map(str, vector.tolist()))

class Read_numpy_arr(FunctionApplier):
    def __init__(self, dtype):
        self.dtype = dtype

    def function_to_apply(self, row):
        return np.fromstring(row, sep=" ", dtype=self.dtype)

class Generate_unique_word_list(FunctionApplier):
    def __init__(self):
        self.unique_words = Counter()
        self.n_t = []
        self.n = []

    def function_to_apply(self, row):
        words = row["content"]
        if type(words) is not list:
            words = literal_eval(words)
        set = row["set"] if "set" in row else 0
        if len(self.n_t) <= set:
            self.n_t.append({})
            self.n.append(0)
            return self.function_to_apply(row)
        n_t = self.n_t[set]
        self.unique_words.update(words)
        self.n[set] += 1
        for word in words:
            if word in n_t:
                n_t[word] += 1
            else:
                n_t[word] = 1
        


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
        return self.unique_words.most_common(nwords)

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


class Tokenizer(FunctionApplier):
    def function_to_apply(self, cell):
        return cell.split()

class Untokenizer(FunctionApplier):
    def function_to_apply(self, cell):
        # Join tokens back into text
        return " ".join(cell)

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
    def __init__(self, remove_punct=True):
        # Create a list of patterns to remove.
        # Compile the patterns to speed up the process
        self.patterns = {
            re.compile(r'(<.*?>)'): '', # remove html tags
            re.compile(r'[<>]'): '', # remove < and >
            re.compile(r'((https?:\/\/)?(?:www\.)?[a-zA-Z0-9-_\+=.:~@#%]+\.[a-zA-Z0-9()]{1,6}\b(?:[a-zA-Z0-9-_.:\\/@#$%&()=+~?]*))'): ' <URL> ', # replace urls with <URL>
            re.compile(r'(https?:\/\/)?w{0,3}\.?[a-z]+\.[a-z]\w*[\w\/-]*'): ' <URL> ', # replace urls with <URL>
            re.compile(r'(\d{1,2}([\:\-/\\]|(,\s)?)){2}\d{2,4}|\d{2,4}(([\:\-/\\]|(,\s)?)\d{1,2}){2}'): ' <DATE> ', # replace dates with <DATE>
            re.compile(r'([Jj]an(uary)?|[Ff]eb(ruary)?|[Mm]ar(ch)?|[Aa]pr(il)?|[Mm]ay|[Jj]un(e)?|[Jj]ul(y)?|[Aa]ug(ust)?|[Ss]ep(tember)?|[Oo]ct(ober)?|[Nn]ov(ember)?|[Dd]ec(ember)?)([\:\-/\\]|(,\s)?)\d{1,2}([\:\-/\\]|(,\s)?)\d{1,4}'): ' <DATE> ', # replace dates with <DATE>
            re.compile(r'([\w.\-]+@(?:[\w-]+\.)+[\w-]{2,4})|@[\w\d]+'): ' <EMAIL> ', # replace email addresses with <EMAIL>
            re.compile(r'(\r\n|\n|\r)+'): ' ', # remove new lines
            re.compile(r'(\t+)'): ' ', # remove tabs
            re.compile(r'(\?)'): ' ? ', # add space before and after question mark
            re.compile(r'(\!)'): ' ! ', # add space before and after exclamation mark
            re.compile(r'[^A-Za-z0-9\s<>\?\!]' if remove_punct else r'[^A-Za-z0-9\s<>\?!\.,]'): '', # remove all special characters, including non-ascii characters and punctuation if remove_punct is True
            re.compile(r'(\d+)(th)?'): ' <NUM> ', # replace numbers with <NUM>
            re.compile(r'( +)'): ' ', # remove multiple spaces
        }

    def function_to_apply(self, cell):
        # Apply patterns using list comprehension
        cell = str(cell)
        cell = cell.lower()
        # Loop through each pattern and apply the pattern to each row and do replacement if needed
        for pattern, replacement in self.patterns.items():
            cell = re.sub(pattern, replacement, cell)

        return cell

class Valid_row(FunctionApplier):
    def __init__(self):
        self.types = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 'bias', 
                        'satire', 'state', 'reliable', 'clickbait', 'political']
    def function_to_apply(self, row):
        # Remove rows which have empty content or start with 'ERROR':
        if row['content'] == '' or row['content'].startswith('ERROR') or row['type'] not in self.types:
            return DELETE_TOKEN

class Join_str_columns(FunctionApplier):
    def __init__(self, columns):
        self.columns = columns
    def function_to_apply(self, row):
        combined = " ".join([row[col] for col in self.columns if type(row[col]) is str]).strip()
        return combined if combined != "" else " " # to avoid nan

#TODO: Change code coppied from Oliver and Daniel
class Clean_author(FunctionApplier):
    def __init__(self):
        self.regex_oddcharacters = re.compile(r'[^A-Za-z0-9\s]')

    def function_to_apply(self, authors):
        author_list = authors.strip().split(",") if type(authors) is str else []
        author_list = [author.strip() for author in author_list]
        author_list = [author.lower() for author in author_list]
        author_list = [(re.sub(self.regex_oddcharacters, "", author)) for author in author_list]
        author_list = "".join(author_list)
        if author_list == "": # to avoid nan
            return " "
        return author_list

class Clean_domain(FunctionApplier):
    def __init__(self):
        self.regex_oddcharacters = re.compile(r'[^A-Za-z0-9\s]')

    def function_to_apply(self, domain):
        domain = domain.split(".")[0]
        domain = domain.lower()
        domain = re.sub(self.regex_oddcharacters, "", domain)
        return domain

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

    def function_to_apply(self, r):
        if not self.has_printed:
            self.has_printed = True
            print(r)


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
            function, to_col = f
            if to_col is None:
                if progress_bar:
                    chunk = chunk.progress_apply(function.function_to_apply, axis=1)
                else:
                    chunk = chunk.apply(function.function_to_apply, axis=1)
            else:
                if progress_bar:
                    chunk[to_col] = chunk[to_col].progress_apply(function.function_to_apply)
                else:
                    chunk[to_col] = chunk[to_col].apply(function.function_to_apply)
                chunk = chunk[chunk[to_col] != DELETE_TOKEN]
        elif len(f) == 3:
            function, from_col, to_col = f
            if from_col is None:
                if progress_bar:
                    chunk[to_col] = chunk.progress_apply(function.function_to_apply, axis=1)
                else:
                    chunk[to_col] = chunk.apply(function.function_to_apply, axis=1)
            else:
                if progress_bar:
                    chunk[to_col] = chunk[from_col].progress_apply(function.function_to_apply)
                else:
                    chunk[to_col] = chunk[from_col].apply(function.function_to_apply)
                chunk = chunk[chunk[to_col] != DELETE_TOKEN]

    # delete rows equal to DELETE_TOKEN
   
    return chunk


def apply_pipeline_pd(df, function_cols):
    # Iterate through each row in the DataFrame and apply the functions
    return applier(function_cols, df.copy())


def apply_pipeline_pd_tqdm(df, function_cols):
    # Iterate through each row in the DataFrame and apply the functions
    return applier(function_cols, df.copy(), progress_bar=True)

def apply_pipeline(old_file, function_cols, new_file=None, batch_size=ROWS_PR_ITERATION, get_batch=False, progress_bar=False, nrows=None):
    i = 0
    start_time = time()

    # Use Pandas chunksize and iterator to read the input file in batches
    with pd.read_csv(old_file, chunksize=batch_size, encoding='utf-8', nrows=nrows) as reader:
        for chunk in reader:
            if function_cols is None:
                return chunk
            # Apply the specified functions to each row in the batch
            chunk = applier(function_cols, chunk, progress_bar=progress_bar)
            
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


def ist_pipeline(srcFile):
    stopwords_lst = stopwords.words('english') 
    # + ["<NUM>", "<DATE>", "<URL>"]
    apply_pipeline(srcFile, [
        (Clean_data(), "content"),
        (Tokenizer(), "content"),
        (Remove_stopwords(stopwords_lst), "content"),
        (Stem(), "content"),
    ], 
    new_file="../datasets/sample/news_sample_cleaned_num_100k.csv",
    progress_bar=True,
    batch_size=100000
    )


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
