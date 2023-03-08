from typing import Tuple
from enum import IntEnum
import shutil
import h5py
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm

TQDM_COLOR = 'magenta'
ROWS_PR_ITERATION = 20000
ROWS = 8529853

# https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv


def num_of_cols_csv(filename: str) -> int:
    with open(filename, 'r', encoding='utf8') as f:
        return len(next(csv.reader(f)))


def remove_file(filename: str):
    if os.path.exists(filename):
        os.remove(filename)


def create_directory(dirname: str):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def remove_directory(dirname: str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def create_empty_string_array(cols: int) -> np.ndarray:
    arr = np.zeros((1, cols), dtype=object)
    arr[0] = ["" for x in range(cols)]
    return arr


class Set(IntEnum):
    TRAIN = 0
    VALI = 1
    TEST = 2


def create_randomly_split_array(size: int = 10, split: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> np.ndarray:
    # Create a numpy array of the given size and set all to zeroes
    arr = np.zeros(size, dtype=Set)
    # Determine the indices for the three splits
    split1 = int(size * split[0])
    split2 = int(size * (split[0] + split[1]))
    # Set the values for the three splits
    arr[split1:split2] = Set.VALI
    arr[split2:] = Set.TEST
    # Shuffle the indexes of the array and return
    np.random.shuffle(arr)
    return arr


def csv_to_hdf(csv_filename: str, hdf_filename: str, cols: int = 0, rows_pr_iteration: int = ROWS_PR_ITERATION):
    with h5py.File(hdf_filename, 'w') as store:
        data_set = store.create_dataset('data', data=create_empty_string_array(cols), maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        # Get and set the header row:
        with open(csv_filename, encoding='utf-8') as f:
            data_set[0] = next(csv.reader(f))
        # Read the rest of the rows and assign to dataset:
        rows = 1
        for c in tqdm(pd.read_csv(csv_filename, encoding='utf-8', dtype=str, chunksize=rows_pr_iteration, lineterminator='\n'),
                      desc='csv to hdf', total=int(ROWS/rows_pr_iteration), unit='rows', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            rows += len(c)
            data_set.resize((rows, cols))
            data_set[-len(c):] = c.astype(str)


def csv_split(csv_filename: str, dirname: str = 'csv-chunks', rows_pr_iteration: int = ROWS_PR_ITERATION, padding: int = 4):
    # Get header row:
    with open(csv_filename, encoding='utf-8') as f:
        header = next(csv.reader(f))
    remove_directory(dirname)
    create_directory(dirname)
    for i, c in tqdm(enumerate(pd.read_csv(csv_filename, encoding='utf-8', dtype=str, chunksize=rows_pr_iteration, lineterminator='\n')),
                     desc='csv split', total=int(ROWS/rows_pr_iteration), unit='splits', colour=TQDM_COLOR):
        df = pd.DataFrame(columns=header)
        pd.concat([df, c], ignore_index=True).to_csv(
            f'{dirname}/{i+1:0{padding}}.csv', index=False)


def create_train_vali_and_test_sets(split: np.ndarray, cols: int, data_filename: str, train_filename: str, vali_filename: str, test_filename: str, rows_pr_iteration: int = ROWS_PR_ITERATION):
    with h5py.File(data_filename, 'r', ) as data,\
            h5py.File(train_filename, 'w', ) as train,\
            h5py.File(vali_filename, 'w', ) as vali,\
            h5py.File(test_filename, 'w', ) as test:
        data_set = data['data']
        arr = create_empty_string_array(cols)
        train_set = train.create_dataset('data', data=arr, maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        vali_set = vali.create_dataset('data', data=arr, maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        test_set = test.create_dataset('data', data=arr, maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        # Set header row:
        train_set[0] = vali_set[0] = test_set[0] = data_set[0, ]
        # Loop through data in chunks and append to the right dataset:
        for start in tqdm(range(1, data_set.shape[0], rows_pr_iteration),
                          desc='create train, vali, and test set', unit='rows', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            end = min(start + rows_pr_iteration, data_set.shape[0])
            chunk_data = data_set[start:end]
            # Get the amount of the split array so it matches the size of the chunk.
            # Split array doesn't have a header row therefore -1.
            chunk_split = split[start-1:end-1]

            # Select the values from the chunk for the train- or vali- or test dataset
            # from the chunk if it matches the shuffled split array:
            train_rows = chunk_data[chunk_split == Set.TRAIN]
            if train_rows.shape[0] > 0:
                train_set.resize(
                    (train_set.shape[0]+train_rows.shape[0], cols))
                train_set[-train_rows.shape[0]:] = train_rows
            vali_rows = chunk_data[chunk_split == Set.VALI]
            if vali_rows.shape[0] > 0:
                vali_set.resize((vali_set.shape[0]+vali_rows.shape[0], cols))
                vali_set[-vali_rows.shape[0]:] = vali_rows
            test_rows = chunk_data[chunk_split == Set.TEST]
            if test_rows.shape[0] > 0:
                test_set.resize((test_set.shape[0]+test_rows.shape[0], cols))
                test_set[-test_rows.shape[0]:] = test_rows


def read_hdf_rows(filename: str, idx: int = 0, num: int = 0):
    with h5py.File(filename, 'r') as f:
        return f['data'][idx:idx+num, ]


def read_hdf_cols(filename: str, idx: int = 0, num: int = 1) -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        return f['data'][:, idx:idx+num]


def num_of_rows_and_cols_hdf(filename: str) -> tuple:
    with h5py.File(filename, 'r', ) as data:
        return data['data'].shape


def run(csv_file: str, hdf_file: str, train_file: str, vali_file: str, test_file: str, rows_pr_iteration: int = ROWS_PR_ITERATION):
    cols = num_of_cols_csv(filename=csv_file)
    csv_to_hdf(csv_filename=csv_file, hdf_filename=hdf_file,
               cols=cols, rows_pr_iteration=rows_pr_iteration)
    rows = num_of_rows_and_cols_hdf(filename=hdf_file)[
        0] - 1  # only rows minus header
    split = create_randomly_split_array(size=rows)
    create_train_vali_and_test_sets(split=split, cols=cols, data_filename=hdf_file, train_filename=train_file,
                                    vali_filename=vali_file, test_filename=test_file, rows_pr_iteration=rows_pr_iteration)


if __name__ == '__main__':
    run(csv_file="../datasets/big/news_cleaned_2018_02_13.csv", hdf_file='../datasets/big/data.h5',
        train_file='../datasets/big/train.h5', vali_file='../datasets/big/vali.h5', test_file='../datasets/big/test.h5')
    #cols = num_of_cols_csv(filename="../datasets/sample/news_sample.csv")
    #csv_to_hdf(csv_filename="../datasets/sample/news_sample.csv", hdf_filename="../datasets/sample/news_sample.h5", cols=cols, rows_pr_iteration=ROWS_PR_ITERATION)
