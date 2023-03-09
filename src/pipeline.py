import h5py
import filehandling as fh
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

class Word_frequency(FunctionApplier):
    def __init__(self, nwords = 50):
        self.swords = nwords
        self.words = []
        self.frequency = Counter()
        self.sorted_frequency = []

    def function_to_apply(self, content):
        # Update/add list of word
        self.frequency.update(content)
        # Return the sorted dictionary based on the frequency of each word
        self.sorted_frequency = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
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
        ps = PorterStemmer()
        stemmed_words = []
        for w in words:
            stemmed_words.append(ps.stem(w))
        return stemmed_words


class Clean_data(FunctionApplier):
    def function_to_apply(self, cell):
        # List of patterns and their appropriate replacements
        patterns = {
            r'(\r\n|\n|\r)+': '(\n)',
            r'( +)': ' ',
            r'(\t+)': '(\t)',
            r'(\!|\[|\])': '',
            r'(\d{1,2}[-/\\]\d{1,2}[-/\\]\d{2,4}|\d{2,4}[-/\\]\d{1,2}[-/\\]\d{1,2})|\w{3}\s\d{1,2}\S\d{4}|\d{1,2}\s\w{3}\s\d{4}|(?:jan(?:uary)?|feb(?:ruary)|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?),? \d{2,4},? \d{2,4}|\d{2,4},? (?:jan(?:uary)?|feb(?:ruary)|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?),? \d{2,4}': '<DATE>',
            r'([\w.\-]+@(?:[\w-]+\.)+[\w-]{2,4})': '<EMAIL>',
            r'((https?:\/\/)?(?:www\.)?[a-zA-Z0-9-_\+=.:~@#%]+\.[a-zA-Z0-9()]{1,6}\b(?:[a-zA-Z0-9-_.:\\/@#$%&()=+~?]*))': '<URL>',
            r'(\.|\,|\?|\&|\"|\”|\“|\%|\:|\-|\(|\)|\´|\`|\’|\$|\'|\|)': '',
            r'(\–|\—)': ' ',
            r'(\d+)(th)?': '<NUM>',
        }
        #print(cell)
        # Convert all to text and lowercase all characters
        cell = cell.lower()
        # Loop through each pattern and apply the pattern to each row and do replacement if needed
        for pattern, replacement in patterns.items():
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

ROWS_PR_ITERATION = 20000
TEST_NUM = 100000
ROWS = 8529853
TQDM_COLOR = 'magenta'

def apply_pipeline(old_file, functions, column=None, new_file=None, rows_pr_iteration=ROWS_PR_ITERATION):
    with h5py.File(old_file, 'r') as f:
        data_set = f['data']
        if new_file is not None:
            save_to = h5py.File(new_file, 'w')
            arr = fh.create_empty_string_array(data_set.shape[1]) 
            save_set = save_to.create_dataset('data', data=arr, maxshape=(data_set.shape), dtype=h5py.string_dtype(encoding='utf-8'))
            save_set.resize((TEST_NUM+1,data_set.shape[1]))#save_set.resize(data_set.shape)
            save_set[0] = data_set[0]
        start_time = time()
        for start in tqdm(range(1, TEST_NUM, rows_pr_iteration),
                      desc='csv to hdf', total=int(ROWS/rows_pr_iteration), unit='rows', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):#data_set.shape[0]
            end = min(start + rows_pr_iteration, data_set.shape[0])
            rows = data_set[start:end]
            for i in range(len(rows)):
                for j, func in enumerate(functions):
                    if column == None:
                        rows[i,column] = func.function_to_apply(rows[i,column])
                    else:
                        if j == 0:
                            acc_out = rows[i,column]
                        acc_out = func.function_to_apply(acc_out)
                        if j == len(functions)-1:
                            rows[i,column] = acc_out
            if new_file is not None:
                save_set[start:end] = rows
        print(f'finish time: {time()-start_time}')
        try:
            save_to.close()
        except:
            pass


# print("INITIAL FILE")
# apply_pipeline("../datasets/sample/data.h5", [Print_content(), Clean_data()], "../datasets/sample/data_cleaned.h5")
# apply_pipeline("../datasets/sample/data.h5", [Print_content(), Clean_data()],"../datasets/sample/data_cleaned.h5")
# print("CLEANED FILE")

stopwords_lst = stopwords.words('english') + ["<NUM>","<DATE>","<URL>"]

apply_pipeline("../datasets/big/data.h5",[
    Decode_to_str(), 
    Clean_data(),
#    Tokenizer(),
#    Remove_stopwords(stopwords_lst),
#    Stem(),
#    Encode_to_json(),
], column=headers["content"], new_file="../datasets/big/data_cleaned.h5")
#apply_pipeline("../datasets/sample/news_sample_cleaned.h5",[Print_first_row()], column=headers["content"])
# apply_pipeline("../datasets/sample/news_sample_cleaned.h5", [Print_content_to_csv(
#    100, "../datasets/sample/out_cleaned.csv")], "../datasets/sample/out_cleaned.h5")

"""
wf = Word_frequency()
apply_pipeline("../datasets/big/data_cleaned.h5",[
    Decode_from_json(), 
    wf
], column=headers["content"])
wf.plot()
"""