from time import time
import h5py
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm

TQDM_COLOR = 'magenta'
# https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv
# csv_file = "datasets/big/news_cleaned_2018_02_13.csv"
ROWS = 8529853
csv_file = "datasets/sample/news_sample.csv"
#ROWS = 250
hdf_file = 'datasets/big/data.h5'
train_file = 'datasets/big/train.h5'
vali_file = 'datasets/big/vali.h5'
test_file = 'datasets/big/test.h5'

# Set the current directory one level up:
os.chdir("..")


def num_of_cols_csv(filename: str) -> int:
    with open(filename, 'r', encoding='utf8') as f:
        return len(next(csv.reader(f)))


def remove_file(filename: str):
    if os.path.exists(filename):
        os.remove(filename)


def create_empty_string_array(cols: int) -> np.ndarray:
    arr = np.zeros((1, cols), dtype=object)
    arr[0] = ["" for x in range(cols)]
    return arr


def create_randomly_split_array(size: int = 10, split: tuple = (0.8, 0.1, 0.1)) -> np.ndarray:
    # 0: Train
    # 1: Vali
    # 2: Test

    # Create a numpy array of the given size and set all to zeroes
    arr = np.zeros(size, dtype=int)
    # Determine the indices for the three splits
    split1 = int(size * split[0])
    split2 = int(size * (split[0] + split[1]))
    # Set the values for the three splits
    arr[split1:split2] = 1
    arr[split2:] = 2
    # Shuffle the indexes of the array and return
    return np.random.shuffle(arr)


def csv_to_hdf(csv_filename: str, hdf_filename: str, cols: int, chunk_size: int = 10):
    remove_file(hdf_filename)
    with h5py.File(hdf_filename, 'w') as store:
        dset = store.create_dataset('data', data=create_empty_string_array(cols), maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        # Get and set the header row:
        with open(csv_filename, encoding='utf-8') as f:
            dset[0] = next(csv.reader(f))
            # Read the rest of the rows and assign to dataset:
        rows = 1
        for c in tqdm(pd.read_csv(csv_filename, encoding='utf-8', dtype=str, chunksize=chunk_size),
                      desc='.csv to .h5', total=int(ROWS/chunk_size), unit='rows', unit_scale=chunk_size, colour=TQDM_COLOR):
            rows += len(c)
            dset.resize((rows, cols))
            dset[-len(c):] = c.astype(str).values


def read_hdf_rows(filename: str, idx: int = 0, num: int = 0) -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        return f['data'][idx:idx+num, :]


def read_hdf_cols(filename: str, idx: int = 0, num: int = 1) -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        return f['data'][:, idx:idx+num]


def create_train_vali_and_test_sets(split, cols: int, data_filename: str, train_filename: str, vali_filename: str, test_filename: str, chunk_size: int = 10000):
    # Remove existing hdf files:
    remove_file(train_filename)
    remove_file(vali_filename)
    remove_file(test_filename)
    # Open input and output files:
    with h5py.File(data_filename, 'r', ) as data,\
            h5py.File(train_filename, 'w', ) as train,\
            h5py.File(vali_filename, 'w', ) as vali,\
            h5py.File(test_filename, 'w', ) as test:
        data = data['data']
        arr = create_empty_string_array()
        trainset = train.create_dataset('data', data=arr, maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        valiset = vali.create_dataset('data', data=arr, maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        testset = test.create_dataset('data', data=arr, maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        # Set header row:
        trainset[0] = valiset[0] = testset[0] = data[0, ]
        # Loop through data in chunks and append to the right dataset:
        for start in tqdm(range(1, data.shape[0], chunk_size), desc='split dataset', unit='rows', unit_scale=chunk_size, colour=TQDM_COLOR):
            end = min(start + chunk_size, data.shape[0])
            chunk_data = data[start:end]
            # Get the amount of the split array so it matches the size of the chunk.
            # Split array doesn't have a header row therefore -1.
            chunk_split = split[start-1:end-1]

            # Select the values from the chunk for the train- or vali- or test dataset
            # from the chunk if it matches the shuffled split array:
            train_rows = chunk_data[chunk_split == 0]
            trainset.resize((trainset.shape[0]+train_rows.shape[0], cols))
            trainset[-train_rows.shape[0]:] = train_rows
            vali_rows = chunk_data[chunk_split == 1]
            valiset.resize((valiset.shape[0]+vali_rows.shape[0], cols))
            valiset[-vali_rows.shape[0]:] = vali_rows
            test_rows = chunk_data[chunk_split == 2]
            testset.resize((testset.shape[0]+test_rows.shape[0], cols))
            testset[-test_rows.shape[0]:] = test_rows


def num_of_rows_and_cols_hdf(filename: str) -> tuple:
    with h5py.File(filename, 'r', ) as data:
        return data['data'].shape


cols = num_of_cols_csv(filename=csv_file)
csv_to_hdf(csv_filename=csv_file, hdf_filename=hdf_file, cols=cols)
split = create_randomly_split_array(size=ROWS)
create_train_vali_and_test_sets(split=split, cols=cols, data_filename=hdf_file,
                                train_filename=train_file, vali_filename=vali_file, test_filename=test_file)

#rows = read_hdf_rows(filename=hdf_file, idx=0, num=ROWS)
#print(rows[0, 2])

# print(num_of_rows_and_cols_hdf(test_file))
