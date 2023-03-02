import subprocess
from time import time
import h5py
import tables as tb
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm
import re

csv_file = "datasets/news_sample.csv"
# https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv
#csv_file = "datasets/news_cleaned_2018_02_13.csv"
# ROWS = 8528956
# COLS = 17
ROWS = 250
COLS = 16
hdf_file = 'data.h5'
train_file = 'train.h5'
vali_file = 'vali.h5'
test_file = 'test.h5'
CHUNK_SIZE = 100
COL_NAMES = ['', 'id', 'domain', 'type', 'url', 'content', 'scraped_at', 'inserted_at', 'updated_at', 'title',
             'authors', 'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary']  # , 'source']
COL_SIZES = {'x': 300, 'id': 150, 'domain': 50, 'type': 50, 'url': 2000, 'content': 200000, 'scraped_at': 100, 'inserted_at': 100, 'updated_at': 100,
             'title': 400, 'authors': 1500, 'keywords': 5, 'meta_keywords': 40000, 'meta_description': 20000, 'tags': 30000, 'summary': 5, 'source': 5}

# TODO:

# Set the current directory one level up:
os.chdir("..")


def num_rows_and_cols_csv(_csv_file: str):
    cmd = 'Import-Csv ".\\' + _csv_file + '" | Measure-Object'
    result = subprocess.run(
        ['powershell', '-Command', cmd], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return result.stderr


# TODO: , lineterminator='\n')),


def csv_to_hdf(csv_filename: str, hdf_filename: str, chunk_size=CHUNK_SIZE, col_names=COL_NAMES):
    # Remove exiting hdf file:
    if os.path.exists(hdf_filename):
        os.remove(hdf_filename)
    # Read csv as chunks so we don't run out of memory and append to hdf file:
    with h5py.File(hdf_filename, 'w') as store:
        arr = np.zeros((1, len(col_names)), dtype=object)
        arr[0] = col_names
        dset = store.create_dataset('data', data=arr, maxshape=(
            None, COLS), dtype=h5py.string_dtype(encoding='utf-8'))
        row_length = 1
        for chunk in tqdm(pd.read_csv(csv_filename, encoding='utf-8', dtype=str, chunksize=chunk_size),
                          desc='csv to hdf format', total=int(ROWS/chunk_size)):
            dset[0] = chunk.columns
            row_length += len(chunk)
            dset.resize((row_length, COLS))
            dset[-len(chunk):] = chunk.astype(str).values


def get_csv_header(csv_file: str):
    df = pd.read_csv(csv_file, nrows=0)
    return df.columns.tolist()


def read_hdf(filename: str, startIdx=0, stopIdx=0):
    with h5py.File(filename, 'r') as f:
        # Access the dataset you want to read
        return f['data'][startIdx:stopIdx+1, ]


def create_train_vali_and_test_sets(split, data_filename: str, train_filename: str, vali_filename: str, test_filename: str, cols_sizes=COL_SIZES):
    # Remove exiting hdf files:
    if os.path.exists(train_filename):
        os.remove(train_filename)
    if os.path.exists(vali_filename):
        os.remove(vali_filename)
    if os.path.exists(test_filename):
        os.remove(test_filename)
    # Run through data file and match each row with the corresponding shuffled array:
    with pd.HDFStore(train_filename, complib='blosc', complevel=9) as train,\
            pd.HDFStore(vali_filename, complib='blosc', complevel=9) as vali,\
            pd.HDFStore(test_filename, complib='blosc', complevel=9) as test:
        for i in tqdm(range(0, len(split), CHUNK_SIZE),
                      desc='create train-, vali- and test sets', total=len(split)):
            for chunk in pd.read_hdf(data_filename, key='data', start=i, chunksize=min(CHUNK_SIZE, len(split)-i)):
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

# ['', 'id', 'domain', 'type', 'url', 'content', 'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors', 'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary', 'source']
# rows, cols = num_rows_and_cols_csv(csv_file)


# print(get_csv_header(csv_file))
# print(num_rows_and_cols_csv(csv_file))
csv_to_hdf(csv_file, hdf_file)
# , columns_to_return=None):
rows = read_hdf(hdf_file, startIdx=0, stopIdx=256)
print(rows[250, 1])

# split = create_randomly_split_array(ROWS)
#
#

# with h5py.File(hdf_file, 'w') as store:
#     arr = np.zeros((1, len(COL_NAMES)), dtype=object)
#     COL_NAMES = ['', 'id', 'domain', 'type', 'url', 'content', 'scraped_at', 'inserted_at', 'updated_at', 'title',
#                      'authors', 'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary', 'source']
#     arr[0] = COL_NAMES
#     dt = h5py.string_dtype(encoding='utf-8')
#     dset = store.create_dataset(
#         'data', data=arr, maxshape=(None, COLS), dtype=dt)
#     for i in range(0, 10):
#         dset.resize((dset.shape[0]+CHUNK_SIZE, COLS))
#         for j in range(0, CHUNK_SIZE):
#             dset[dset.shape[0]-CHUNK_SIZE+j, :] = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
#                                                    '10', '11', '12', '13', '14', '15', '16', '17']
#     print(dset[100, 1])
# create_train_vali_and_test_sets(split, data_filename=hdf_file, train_filename=train_file, vali_filename = vali_file, test_filename = test_file)
