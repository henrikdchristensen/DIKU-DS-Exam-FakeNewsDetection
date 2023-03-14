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


def create_randomly_array(old_size: int = 10, new_size: int = 5) -> np.ndarray:
    # Create a numpy array of the given size and set all to zeroes
    arr = np.zeros(old_size, dtype=bool)
    # Determine the indices for the three splits
    arr[0:new_size] = True
    # Shuffle the indexes of the array and return
    np.random.shuffle(arr)
    return arr

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


def csv_split(filename: str, dirname: str = 'csv-chunks', rows_pr_iteration: int = ROWS_PR_ITERATION, padding: int = 4):
    # Get header row:
    with open(filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    remove_directory(dirname)
    create_directory(dirname)
    for i, c in tqdm(enumerate(pd.read_csv(filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n')),
                     desc='csv split', unit='splits', colour=TQDM_COLOR):
        pd.concat([colnames, c], ignore_index=True).to_csv(
            f'{dirname}/{i+1:0{padding}}.csv', index=False)


def number_of_rows(filename: str, rows_pr_iteration: int) -> int:
    rows = 0
    with pd.read_csv(filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n') as reader:
        for chunk in tqdm(reader, desc='counting rows', unit='chunks', colour=TQDM_COLOR):
            rows += chunk.shape[0]
    print(f"number of rows in {filename}: {rows}")
    return rows

def random_dataset(orig_rows: int, new_rows: int, in_filename: str, out_filename: str, rows_pr_iteration: int):
    if rows_pr_iteration > orig_rows:
        rows_pr_iteration = orig_rows
    samples = create_randomly_array(old_size=orig_rows, new_size=new_rows)
    with open(in_filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    colnames.to_csv(out_filename, mode='w')
    # Loop through data in chunks and append to the right dataset:
    start = 0
    with pd.read_csv(in_filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n', nrows=orig_rows) as reader:
        for chunk in tqdm(reader, desc='sampling the dataset randomly', total=int(orig_rows/rows_pr_iteration), unit='rows', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            end = min(start + rows_pr_iteration, orig_rows)
            # Get the amount of the split array so it matches the size of the chunk.
            chunk_split = samples[start:end]
            start += chunk.shape[0]
            # Select the values from the chunk for the train- or vali- or test dataset
            # from the chunk if it matches the shuffled split array:
            rows = chunk[chunk_split == True]
            if rows.shape[0] > 0:
                rows.to_csv(out_filename, mode='a', header=None)


def create_train_vali_and_test_sets(rows: int = 10, split: Tuple[float, float, float] = (0.8, 0.1, 0.1), data_filename: str = '', train_filename: str = '', vali_filename: str = '', test_filename: str = '', rows_pr_iteration: int = ROWS_PR_ITERATION):
    if rows_pr_iteration > rows:
        rows_pr_iteration = rows
    split = create_randomly_split_array(size=rows, split=split)
    with open(data_filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    colnames.to_csv(train_filename, mode='w')
    colnames.to_csv(vali_filename, mode='w')
    colnames.to_csv(test_filename, mode='w')
    # Loop through data in chunks and append to the right dataset:
    start = 0
    with pd.read_csv(data_filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n', nrows=rows) as reader:
        for chunk in tqdm(reader, desc='splitting data into: train-, vali-, and test set', total=int(rows/rows_pr_iteration), unit='rows', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
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


def run(data_filename: str, train_filename: str, vali_filename: str, test_filename: str, rows_pr_iteration: int = ROWS_PR_ITERATION):
    rows = number_of_rows(filename=data_filename, rows_pr_iteration=rows_pr_iteration)
    new_dataset = '../datasets/sample/new.csv'
    size_of_new_dataset = 200
    random_dataset(orig_rows=rows, new_rows=size_of_new_dataset, in_filename=data_filename, out_filename=new_dataset, rows_pr_iteration=rows_pr_iteration)
    create_train_vali_and_test_sets(rows=size_of_new_dataset, split=(0.8, 0.1, 0.1), data_filename=new_dataset, train_filename=train_filename, vali_filename=vali_filename, test_filename=test_filename, rows_pr_iteration=rows_pr_iteration)


if __name__ == '__main__':
    run(data_filename="../datasets/sample/news_sample.csv", train_filename='../datasets/sample/train.csv', vali_filename='../datasets/sample/vali.csv', test_filename='../datasets/sample/test.csv')
    #run(csv_file="../datasets/big/news_cleaned_2018_02_13.csv", train_file='../datasets/big/train.csv', vali_file='../datasets/big/vali.csv', test_file='../datasets/big/test.csv', rows_pr_iteration=ROWS_PR_ITERATION)