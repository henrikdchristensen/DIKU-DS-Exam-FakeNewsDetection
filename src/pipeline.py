#import filehandling as fh
import csv
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
import filehandling as fh
from textblob import TextBlob

tqdm.pandas()


labels: dict = {
    'fake': False,
    'conspiracy': False,
    'junksci': False,
    'hate': False,
    'unreliable': False,
    'bias': False,
    'satire': False,
    # 'state': False,
    'reliable': True,
    'clickbait': True,
    'political': True
}

ROWS_PR_ITERATION = 20000
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


def get_dataframe_with_distribution(file, total_size, splits, balanced, end_col="set", type_col="type",  chunksize=ROWS_PR_ITERATION, out_file=None, get_frame=True, classes=labels, delete=True):
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
                if get_frame:
                    return data
                return
    print("ERROR: not enough data to create sets")
    return data


class Debug(FunctionApplier):
    def __init__(self):
        self.i = 0

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
        if content_lst == []:  # to avoid nan
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


class Remove_stopwords(FunctionApplier):
    def __init__(self, swords):
        self.swords = swords

    def function_to_apply(self, words):
        return [w for w in words if not w in self.swords]


class Stem(FunctionApplier):
    def function_to_apply(self, words):
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
            re.compile(r'(<.*?>)'): '',  # remove html tags
            re.compile(r'[<>]'): '',  # remove < and >
            # replace urls with <URL>
            re.compile(r'((https?:\/\/)?(?:www\.)?[a-zA-Z0-9-_\+=.:~@#%]+\.[a-zA-Z0-9()]{1,6}\b(?:[a-zA-Z0-9-_.:\\/@#$%&()=+~?]*))'): ' <URL> ',
            re.compile(r'(https?:\/\/)?w{0,3}\.?[a-z]+\.[a-z]\w*[\w\/-]*'): ' <URL> ',  # replace urls with <URL>
            # replace dates with <DATE>
            re.compile(r'(\d{1,2}([\:\-/\\]|(,\s)?)){2}\d{2,4}|\d{2,4}(([\:\-/\\]|(,\s)?)\d{1,2}){2}'): ' <DATE> ',
            # replace dates with <DATE>
            re.compile(r'([Jj]an(uary)?|[Ff]eb(ruary)?|[Mm]ar(ch)?|[Aa]pr(il)?|[Mm]ay|[Jj]un(e)?|[Jj]ul(y)?|[Aa]ug(ust)?|[Ss]ep(tember)?|[Oo]ct(ober)?|[Nn]ov(ember)?|[Dd]ec(ember)?)([\:\-/\\]|(,\s)?)\d{1,2}([\:\-/\\]|(,\s)?)\d{1,4}'): ' <DATE> ',
            # replace email addresses with <EMAIL>
            re.compile(r'([\w.\-]+@(?:[\w-]+\.)+[\w-]{2,4})|@[\w\d]+'): ' <EMAIL> ',
            re.compile(r'(\r\n|\n|\r)+'): ' ',  # remove new lines
            re.compile(r'(\t+)'): ' ',  # remove tabs
            re.compile(r'(\?)'): ' ? ',  # add space before and after question mark
            re.compile(r'(\!)'): ' ! ',  # add space before and after exclamation mark
            re.compile(r'(\-)'): ' ',
            # remove all special characters, including non-ascii characters and punctuation if remove_punct is True
            re.compile(r'[^A-Za-z0-9\s<>\?\!]' if remove_punct else r'[^A-Za-z0-9\s<>\?!\.,]'): '',
            re.compile(r'(\d+)(th)?'): ' <NUM> ',  # replace numbers with <NUM>
            re.compile(r'( +)'): ' ',  # remove multiple spaces
        }

    def function_to_apply(self, cell):
        # Apply patterns using list comprehension
        cell = str(cell)
        cell = cell.lower()
        # Loop through each pattern and apply the pattern to each row and do replacement if needed
        for pattern, replacement in self.patterns.items():
            cell = re.sub(pattern, replacement, cell)

        return cell


class Join_str_columns(FunctionApplier):
    def __init__(self, columns):
        self.columns = columns

    def function_to_apply(self, row):
        combined = " ".join([row[col] for col in self.columns if type(row[col]) is str]).strip()
        return combined if combined != "" else " "  # to avoid nan

# TODO: Change code coppied from Oliver and Daniel


class Clean_author(FunctionApplier):
    def __init__(self):
        self.regex_oddcharacters = re.compile(r'[^A-Za-z0-9\s]')

    def function_to_apply(self, authors):
        author_list = authors.strip().split(",") if type(authors) is str else []
        author_list = [author.strip() for author in author_list]
        author_list = [author.lower() for author in author_list]
        author_list = [(re.sub(self.regex_oddcharacters, "", author)) for author in author_list]
        author_list = "".join(author_list)
        if author_list == "":  # to avoid nan
            return ""
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


class Binary_labels_LIAR(FunctionApplier):
    def __init__(self):
        self.binary_labels: dict = {
            'pants-fire': False,
            'false': False,
            'barely-true': False,
            'half-true': True,
            'mostly-true': True,
            'True': True
        }

    def function_to_apply(self, cell):
        try:
            binary_label = self.binary_labels[cell]
        except:
            binary_label = True
        return binary_label


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


class Sentence_analysis(FunctionApplier):
    def function_to_apply(self, cell):
        return (TextBlob(str(cell)).sentiment.polarity, TextBlob(str(cell)).sentiment.subjectivity)


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
                    chunk = chunk[chunk['content'] != DELETE_TOKEN]
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
    return chunk


def apply_pipeline_pd(df, function_cols):
    # Iterate through each row in the DataFrame and apply the functions
    return applier(function_cols, df.copy())


def apply_pipeline_pd_tqdm(df, function_cols):
    # Iterate through each row in the DataFrame and apply the functions
    return applier(function_cols, df.copy(), progress_bar=True)


def apply_pipeline(old_file, function_cols, new_file=None, batch_size=ROWS_PR_ITERATION, get_batch=False, progress_bar=True, total_rows=20000):
    i = 0
    start_time = time()

    # Use Pandas chunksize and iterator to read the input file in batches
    with pd.read_csv(old_file, chunksize=batch_size, encoding='utf-8', nrows=total_rows) as reader:
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

            i += len(chunk)
            print(f'processed {i} rows')
        # Print the time taken to process the data
        print(f'finish time: {time()-start_time}')


class Remove_unwanted_rows_and_cols():
    def __init__(self, filename, new_filename):
        self.headers_to_keep = {
            'id': True,
            'domain': True,
            'type': True,
            'url': False,
            'content': True,
            'scraped_at': False,
            'inserted_at': False,
            'updated_at': False,
            'title': True,
            'authors': True,
            'keywords': False,
            'meta_keywords': False,
            'meta_description': False,
            'tags': False,
            'summary\r': False,
        }

        self.labels: dict = {
            'fake': False,
            'conspiracy': False,
            'junksci': False,
            'hate': False,
            'unreliable': False,
            'bias': False,
            'satire': False,
            # 'state': False,
            'reliable': True,
            'clickbait': True,
            'political': True
        }
        self.filename = filename
        self.new_filename = new_filename

    def run(self):
        # Remove output file if it already exists
        fh.remove_file(self.new_filename)
        # Create a file for each chunk in the directory:
        first_iteration = True
        for c in tqdm(pd.read_csv(self.filename, encoding='utf-8', chunksize=ROWS_PR_ITERATION, lineterminator='\n'),
                      desc='remove unwanted rows and cols', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            # Remove columns which are not True in self.headers_to_keep:
            c = c[[k for k, v in self.headers_to_keep.items() if v]]
            # Remove rows which have empty content or start with 'ERROR' or have a type not in self.labels or have a nan domain:
            c = c[c['content'].notna() & ~c['content'].str.startswith('Error') &
                  c['type'].isin(self.labels.keys()) & c['domain'].notna()]
            c.to_csv(self.new_filename, index=False, mode='a', header=first_iteration)
            first_iteration = False


def simple_model_test():
    sm = Simple_model()
    apply_pipeline("../datasets/big/news_sample_cleaned.csv", [
        (sm, None)
    ], )
    sm.get_metrics()


def create_dataset(file, unwanted_removed_file, cleaned_file, cleaned_file_combined):
    Remove_unwanted_rows_and_cols(file, unwanted_removed_file).run()

    stopwords_lst = stopwords.words('english')
    apply_pipeline(unwanted_removed_file, [
        (Binary_labels(), 'type', 'type_binary'),

        (Clean_domain(), 'domain'),

        (Clean_author(), "authors"),

        (Clean_data(), 'content', 'content_cleaned'),
        (Tokenizer(), "content_cleaned"),
        (Stem(), "content_cleaned"),
        (Combine_Content(), "content_cleaned", "content_combined"),
        (Remove_stopwords(stopwords_lst), "content_cleaned"),

        (Clean_data(), 'title'),
        (Tokenizer(), "title"),
        (Remove_stopwords(stopwords_lst), "title"),
        (Stem(), "title"),
        (Combine_Content(), "title"),

        (Sentence_analysis(), "content_combined", "sentence_analysis"),
    ],
        new_file=cleaned_file,
        progress_bar=True,
    )

    apply_pipeline(cleaned_file, [
        (Join_str_columns(
            ['content_combined', 'authors']), None, 'content_authors'),
        (Join_str_columns(
            ['content_combined', 'title']), None, 'content_title'),
        (Join_str_columns(
            ['content_combined', 'domain']), None, 'content_domain'),
        (Join_str_columns(['content_combined', 'domain',
                           'authors', 'title']), None, 'content_domain_authors_title')
    ],
        new_file=cleaned_file_combined,
        progress_bar=True,
    )


def run():
    choice = input("Press 's' for sample or 'l' for large dataset or 'x' to Exit: ")
    if choice == 'x':
        print("exiting")
        return
    elif choice == 's':
        path = "../datasets/sample/"
    elif choice == 'l':
        path = "../datasets/large/"
    else:
        print("Invalid choice - exiting")
        return
    create_dataset(file=path+"shuffled.csv", unwanted_removed_file=path +
                   "unwanted_removed.csv", cleaned_file=path+"dataset.csv", cleaned_file_combined=path+"dataset_combined.csv")


if __name__ == '__main__':
    run()
