import subprocess
from time import time
import h5py
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm

TQDM_COLOR = 'magenta'
# https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv
csv_file = "datasets/news_cleaned_2018_02_13.csv"
ROWS = 8528956
COLS = 17
#csv_file = "datasets/news_sample.csv"
#ROWS = 250
#COLS = 16
hdf_file = 'datasets/data.h5'
train_file = 'datasets/train.h5'
vali_file = 'datasets/vali.h5'
test_file = 'datasets/test.h5'

# Set the current directory one level up:
os.chdir("..")


def num_rows_and_cols_csv(csv_file: str):
    # On MacOS: Use another command:
    cmd = 'Import-Csv ".\\' + csv_file + '" | Measure-Object'
    result = subprocess.run(
        ['powershell', '-Command', cmd], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return result.stderr


def remove_file(filename: str):
    if os.path.exists(filename):
        os.remove(filename)


def empty_string_array():
    arr = np.zeros((1, COLS), dtype=object)
    arr[0] = ["" for x in range(COLS)]
    return arr


def csv_to_hdf(csv_filename: str, hdf_filename: str, chunk_size: int = 10):
    remove_file(hdf_filename)
    with h5py.File(hdf_filename, 'w') as store:
        dset = store.create_dataset('data', data=empty_string_array(), maxshape=(
            None, COLS), dtype=h5py.string_dtype(encoding='utf-8'))
        # Get and set the header row:
        with open(csv_filename, encoding='utf-8') as f:
            dset[0] = next(csv.reader(f))
            # Read the rest of the rows and assign to dataset:
        rows = 1
        for c in tqdm(pd.read_csv(csv_filename, encoding='utf-8', dtype=str, chunksize=chunk_size),
                      desc='.csv to .h5', total=int(ROWS/chunk_size), unit='rows', unit_scale=chunk_size, colour=TQDM_COLOR):
            rows += len(c)
            dset.resize((rows, COLS))
            dset[-len(c):] = c.astype(str).values


def read_hdf_rows(filename: str, idx=0, num=0):
    with h5py.File(filename, 'r') as f:
        return f['data'][idx:idx+num, :]


def read_hdf_cols(filename: str, idx=0, num=1):
    with h5py.File(filename, 'r') as f:
        return f['data'][:, idx:idx+num]


def create_train_vali_and_test_sets(split, data_filename: str, train_filename: str, vali_filename: str, test_filename: str, chunk_size: int = 10000):
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
        arr = empty_string_array()
        trainset = train.create_dataset('data', data=arr, maxshape=(
            None, COLS), dtype=h5py.string_dtype(encoding='utf-8'))
        valiset = vali.create_dataset('data', data=arr, maxshape=(
            None, COLS), dtype=h5py.string_dtype(encoding='utf-8'))
        testset = test.create_dataset('data', data=arr, maxshape=(
            None, COLS), dtype=h5py.string_dtype(encoding='utf-8'))
        # Set header row:
        trainset[0] = valiset[0] = testset[0] = data[0, ]
        # Loop through data in chunks and append to the right dataset:
        for start in tqdm(range(1, data.shape[0], chunk_size), desc='split dataset', unit='rows', unit_scale=chunk_size, colour=TQDM_COLOR):
            end = min(start + chunk_size, data.shape[0])
            chunk = data[start:end]
            chunk_split = split[start-1:end-1]
            train_rows = chunk[chunk_split == 0]
            vali_rows = chunk[chunk_split == 1]
            test_rows = chunk[chunk_split == 2]
            trainset.resize((trainset.shape[0]+train_rows.shape[0], COLS))
            trainset[-train_rows.shape[0]:] = train_rows
            valiset.resize((valiset.shape[0]+vali_rows.shape[0], COLS))
            valiset[-vali_rows.shape[0]:] = vali_rows
            testset.resize((testset.shape[0]+test_rows.shape[0], COLS))
            testset[-test_rows.shape[0]:] = test_rows


def create_randomly_split_array(size: int = 1):
    # 0: Training
    # 1: Validation
    # 2: Test

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


# print(num_rows_and_cols_csv(csv_file))
#csv_to_hdf(csv_file, hdf_file)

rows = read_hdf_rows(filename=hdf_file, idx=0, num=256)
print(rows[0, 2])

split = create_randomly_split_array(size=ROWS)
create_train_vali_and_test_sets(split, data_filename=hdf_file,
                                train_filename=train_file, vali_filename=vali_file, test_filename=test_file)
