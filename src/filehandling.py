from typing import Tuple
from enum import IntEnum
import shutil
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm

TQDM_COLOR = 'magenta'
ROWS_PR_ITERATION = 20000
ROWS = 8528956

#ROWS_PR_ITERATION = 20
#ROWS = 250

# https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv


def remove_file(filename: str):
    if os.path.exists(filename):
        os.remove(filename)


def create_directory(dirname: str):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def remove_directory(dirname: str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


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


def csv_split(csv_filename: str, dirname: str = 'csv-chunks', rows_pr_iteration: int = ROWS_PR_ITERATION, padding: int = 4):
    # Get header row:
    with open(csv_filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    remove_directory(dirname)
    create_directory(dirname)
    for i, c in tqdm(enumerate(pd.read_csv(csv_filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n')),
                     desc='csv split', total=int(ROWS/rows_pr_iteration), unit='splits', colour=TQDM_COLOR):
        pd.concat([colnames, c], ignore_index=True).to_csv(
            f'{dirname}/{i+1:0{padding}}.csv', index=False)


def number_of_rows(filename: str, rows_pr_iteration: int) -> int:
    rows = 0
    with pd.read_csv(filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n') as reader:
        for chunk in tqdm(reader, desc='counting rows', total=ROWS/rows_pr_iteration, unit='rows', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            rows += chunk.shape[0]
    return rows

def create_train_vali_and_test_sets(split: np.array, rows: int, data_filename: str, train_filename: str, vali_filename: str, test_filename: str, rows_pr_iteration: int):
    with open(data_filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    colnames.to_csv(train_filename, mode='w')
    colnames.to_csv(vali_filename, mode='w')
    colnames.to_csv(test_filename, mode='w')
    
    # Loop through data in chunks and append to the right dataset:
    start = 0
    with pd.read_csv(data_filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n') as reader:
        for chunk in tqdm(reader, desc='splitting data into: train-, vali-, and test set', total=rows/rows_pr_iteration, unit='rows', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            
            end = min(start + rows_pr_iteration, rows)
            # Get the amount of the split array so it matches the size of the chunk.
            chunk_split = split[start:end]
            start += chunk.shape[0]
            # Select the values from the chunk for the train- or vali- or test dataset
            # from the chunk if it matches the shuffled split array:
            train_rows = chunk[chunk_split == Set.TRAIN]
            if train_rows.shape[0] > 0:
                train_rows.to_csv(train_filename, mode='a', header=None)
            vali_rows = chunk[chunk_split == Set.VALI]
            if vali_rows.shape[0] > 0:
                vali_rows.to_csv(vali_filename, mode='a', header=None)
            test_rows = chunk[chunk_split == Set.TEST]
            if test_rows.shape[0] > 0:
                test_rows.to_csv(test_filename, mode='a', header=None)


def read_rows(filename: str, idx: int = 0, num: int = 0):
    return pd.read_csv(filename, encoding='utf-8', lineterminator='\n', skiprows=idx, nrows=num)


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
    #run(csv_file="../datasets/sample/news_min.csv", hdf_file='../datasets/sample/data_min.h5',
    #    train_file='../datasets/sample/train_min.h5', vali_file='../datasets/sample/vali_min.h5', test_file='../datasets/sample/test_min.h5')
    #df = read_rows(filename="../datasets/big/data_cleaned.csv", idx=ROWS, num=1)
    arr = create_randomly_split_array(size=ROWS, split=(0.8, 0.1, 0.1))#ROWS
    create_train_vali_and_test_sets(split=arr, rows=ROWS, data_filename="../datasets/big/news_cleaned_2018_02_13.csv", train_filename="../datasets/big/train.csv", vali_filename="../datasets/big/vali.csv", test_filename="../datasets/big/test.csv", rows_pr_iteration=ROWS_PR_ITERATION)
    #print(number_of_rows(filename="../datasets/big/news_cleaned_2018_02_13.csv", rows_pr_iteration=ROWS_PR_ITERATION))
