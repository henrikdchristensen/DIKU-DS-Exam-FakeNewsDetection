import subprocess
from time import time
import h5py
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm


# https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv
csv_file = "datasets/news_cleaned_2018_02_13.csv"
ROWS = 8528956
COLS = 17
#csv_file = "datasets/news_sample.csv"
#ROWS = 250
#COLS = 16
hdf_file = 'data.h5'
train_file = 'train.h5'
vali_file = 'vali.h5'
test_file = 'test.h5'
CHUNK_SIZE = 10

# Set the current directory one level up:
os.chdir("..")


def num_rows_and_cols_csv(_csv_file: str):
    # On MacOS: Use another command:
    cmd = 'Import-Csv ".\\' + _csv_file + '" | Measure-Object'
    result = subprocess.run(
        ['powershell', '-Command', cmd], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return result.stderr


def remove_file(filename: str):
    if os.path.exists(filename):
        os.remove(filename)


def csv_to_hdf(csv_filename: str, hdf_filename: str):

    remove_file(hdf_filename)

    # Read csv as chunks so we don't run out of memory and append to hdf file:
    with h5py.File(hdf_filename, 'w') as store:
        # Create dataset with one dummy string array:
        arr = np.zeros((1, COLS), dtype=object)
        arr[0] = ["" for x in range(COLS)]
        dset = store.create_dataset('data', data=arr, maxshape=(
            None, COLS), dtype=h5py.string_dtype(encoding='utf-8'))
        # Get and set the header row:
        with open(csv_filename, encoding='utf-8') as f:
            dset[0] = next(csv.reader(f))
        # Read the rest of the rows and assign to dataset:
        rows = 1
        for c in tqdm(pd.read_csv(csv_filename, encoding='utf-8', dtype=str, chunksize=CHUNK_SIZE),
                      desc='.csv to .h5', total=int(ROWS/CHUNK_SIZE)):
            rows += len(c)
            dset.resize((rows, COLS))
            dset[-len(c):] = c.astype(str).values


def read_hdf(filename: str, startIdx=0, stopIdx=0):
    with h5py.File(filename, 'r') as f:
        return f['data'][startIdx:stopIdx+1, ]


def create_train_vali_and_test_sets(split, data_filename: str, train_filename: str, vali_filename: str, test_filename: str):
    # Remove exiting hdf files:
    remove_file(train_filename)
    remove_file(vali_filename)
    remove_file(test_filename)
    # Run through data file and match each row with the corresponding shuffled array:
    with h5py.File(data_filename, 'r', ) as data,\
            h5py.File(train_filename, 'w') as train,\
            h5py.File(vali_filename, 'w') as vali,\
            h5py.File(test_filename, 'w') as test:

        data = data['data']

        arr = np.zeros((1, COLS), dtype=object)
        arr[0] = ["" for x in range(COLS)]
        trainset = train.create_dataset('train', data=arr, maxshape=(
            None, COLS), dtype=h5py.string_dtype(encoding='utf-8'))
        valiset = vali.create_dataset('vali', data=arr, maxshape=(
            None, COLS), dtype=h5py.string_dtype(encoding='utf-8'))
        testset = test.create_dataset('test', data=arr, maxshape=(
            None, COLS), dtype=h5py.string_dtype(encoding='utf-8'))

        # Set header row:
        trainset[0] = valiset[0] = testset[0] = data[0, ]
        for i in tqdm(range(0, len(split)),
                      desc='split dataset', total=len(split)):
            match split[i]:
                case 0:
                    trainset.resize((trainset.shape[0]+1, COLS))
                    trainset[-1:] = data[i+1]
                case 1:
                    valiset.resize((valiset.shape[0]+1, COLS))
                    valiset[-1:] = data[i+1]
                case 2:
                    testset.resize((testset.shape[0]+1, COLS))
                    testset[-1:] = data[i+1]


def create_randomly_split_array(size: int):
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
csv_to_hdf(csv_file, hdf_file)

rows = read_hdf(hdf_file, startIdx=0, stopIdx=256)
print(rows[0, 2])

split = create_randomly_split_array(ROWS)
create_train_vali_and_test_sets(split, data_filename=hdf_file,
                                train_filename=train_file, vali_filename=vali_file, test_filename=test_file)
