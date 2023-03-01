# file = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"
# file = "datasets/news_sample.csv"
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm

#csv_file = "datasets/news_sample.csv"
csv_file = "datasets/news_cleaned_2018_02_13.csv"
hdf_file = 'data.h5'

ROWS = 216212648

# Set the current directory one level up:
os.chdir("..")


def blocks(files, size=65536):
    # https://stackoverflow.com/questions/9629179/python-counting-lines-in-a-huge-10gb-file-as-fast-as-possible
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def num_rows_and_cols_csv(csv_file: str):
    with open(csv_file, mode="rb") as f:
        rows = sum(block.count(b"\n") for block in blocks(f))
    with open(csv_file, mode="r", encoding='utf8') as f:
        reader = csv.reader(f)
        cols = len(next(reader))
    return rows, cols


def csv_to_hdf(csv_filename: str, hdf_filename: str, cols_sizes, chunk_size=500):
    # Remove exiting hdf file:
    if os.path.exists(hdf_filename):
        os.remove(hdf_filename)
    # Read csv as chunks and append to hdf file:
    with pd.HDFStore(hdf_filename, complib='blosc', complevel=9) as store:
        for chunk in tqdm(pd.read_csv(csv_filename, chunksize=chunk_size,
                                      names=['x', 'id', 'domain', 'type', 'url', 'content', 'scraped_at', 'inserted_at', 'updated_at', 'title',
                                             'authors', 'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary', 'source']), total=ROWS/chunk_size):
            chunk = chunk.astype(str)  # TODO: Not sure if it's needed...
            store.append(key='data', value=chunk,
                         index=False, min_itemsize=cols_sizes)


def get_csv_header(csv_file: str):
    with open(csv_file, mode='r', encoding="utf-8") as f:
        return next(csv.reader(f))


def read_hdf(filename: str, startIdx=0, stopIdx=0, columns_to_return=None):
    return pd.read_hdf(filename, mode='r', start=startIdx, stop=stopIdx, columns=columns_to_return, iterator=False, chunksize=1000)


def create_train_vali_and_test_sets(split, data_filename: str, train_filename: str, vali_filename: str, test_filename: str, cols_sizes):
    if os.path.exists(train_filename):
        os.remove(train_filename)
    if os.path.exists(vali_filename):
        os.remove(vali_filename)
    if os.path.exists(test_filename):
        os.remove(test_filename)
    with pd.HDFStore(train_filename, complib='blosc', complevel=9) as train, pd.HDFStore(vali_filename, complib='blosc', complevel=9) as vali, pd.HDFStore(test_filename, complib='blosc', complevel=9) as test:
        for i in tqdm(range(0, len(split), 500), total=len(split)):
            for j, chunk in pd.read_hdf(data_filename, key='data', start=i, chunksize=500):
                match split[i]:
                    case 0: train.append(key='train', value=chunk,
                                         index=False, min_itemsize=cols_sizes)
                    case 1: vali.append(key='vali', value=chunk,
                                        index=False, min_itemsize=cols_sizes)
                    case 2: test.append(key='test', value=chunk,
                                        index=False, min_itemsize=cols_sizes)


def create_randomly_split_array(size: int):
    # Create a numpy array of the given size and set all to zeroes
    arr = np.zeros(size, dtype=int)

    # Determine the indices for the three splits
    split1 = int(size * 0.8)
    split2 = int(size * 0.9)

    # Set the other two's values
    arr[split1:split2] = 1
    arr[split2:] = 2

    # Shuffle the indexes of the array
    np.random.shuffle(arr)

    return arr


colssizes = {'x': 300, 'id': 150, 'domain': 40, 'type': 5, 'url': 700, 'content': 200000, 'scraped_at': 5, 'inserted_at': 5, 'updated_at': 5,
             'title': 400, 'authors': 800, 'keywords': 5, 'meta_keywords': 40000, 'meta_description': 15000, 'tags': 30000, 'summary': 5, 'source': 5}
#['', 'id', 'domain', 'type', 'url', 'content', 'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors', 'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary', 'source']
#rows, cols = num_rows_and_cols_csv(csv_file)

split = create_randomly_split_array(ROWS)
csv_to_hdf(csv_file, hdf_file, colssizes)
create_train_vali_and_test_sets(split, data_filename=hdf_file, train_filename='train.h5',
                                vali_filename='vali.h5', test_filename='test.h5', cols_sizes=colssizes)
#df = read_hdf(hdf_file)
#value = df.iloc[1, 'domain']
# print(value)

# print(get_csv_header(csv_file))
