from typing import Tuple
from enum import IntEnum
import shutil
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm

TQDM_COLOR = 'magenta'

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
    TRAIN = 1
    VALI = 2
    TEST = 3


def create_randomly_split_array(old_size: int, new_size: int, split: Tuple[float, float, float]) -> np.ndarray:
    assert sum(split) == 1.0, "split values must add up to 1.0"
    # Create a numpy array of the given size and set all to zeroes
    arr = np.zeros(old_size, dtype=Set)
    # Determine the indices for the three splits
    split1 = int(new_size * split[0])
    split2 = int(new_size * (split[0] + split[1]))
    split3 = int(new_size * (split[0] + split[1] + split[2]))
    # Set the values for the three splits
    # index 0 (inclusive) up to index "split1" (exclusive)
    arr[0:split1] = Set.TRAIN
    arr[split1:split2] = Set.VALI
    arr[split2:split3] = Set.TEST
    # Shuffle the indexes of the array and return
    np.random.shuffle(arr)
    return arr


def create_randomly_array(old_size: int, new_size: int) -> np.ndarray:
    # Create a numpy array of the given size and set all to zeroes
    arr = np.zeros(old_size, dtype=bool)
    # Set the values for the three splits
    # index 0 (inclusive) up to index "split1" (exclusive)
    arr[0:new_size] = True
    # Shuffle the indexes of the array and return
    np.random.shuffle(arr)
    return arr


def csv_split(filename: str, dirname: str = 'csv-chunks', rows_pr_iteration: int = 20000, padding: int = 4):
    # Get header row:
    with open(filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    # Remove existing directory and create new:
    remove_directory(dirname)
    # Add each chunk to a file in the directory:
    create_directory(dirname)
    for i, c in tqdm(enumerate(pd.read_csv(filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n')),
                     desc='csv split', unit='splits', colour=TQDM_COLOR):
        pd.concat([colnames, c], ignore_index=True).to_csv(
            f'{dirname}/{i+1:0{padding}}.csv', index=False)


def number_of_rows(filename: str, rows_pr_iteration: int) -> int:
    # Count the number of rows in the file:
    rows = 0
    with pd.read_csv(filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n') as reader:
        for chunk in tqdm(reader, desc='counting rows', unit='chunks', colour=TQDM_COLOR):
            rows += chunk.shape[0]
    print(f"rows in original dataset: {rows}")
    return rows


def create_train_vali_and_test_sets(old_size: int, new_size: int, split: Tuple[float, float, float], data_filename: str, train_filename: str, vali_filename: str, test_filename: str, rows_pr_iteration: int = 20000):
    rows_pr_iteration = min(rows_pr_iteration, old_size)
    new_size = min(new_size, old_size)
    split = create_randomly_split_array(
        old_size=old_size, new_size=new_size, split=split)
    # Write the header row to the new files:
    with open(data_filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    colnames.to_csv(train_filename, mode='w')
    colnames.to_csv(vali_filename, mode='w')
    colnames.to_csv(test_filename, mode='w')
    # Loop through data in chunks and append to the right dataset:
    start = 0
    with pd.read_csv(data_filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n') as reader:
        for chunk in tqdm(reader, desc='splitting data into: train-, vali-, and test set', total=int(old_size/rows_pr_iteration), unit='rows encountered in orig. dataset', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            # Get the amount of the split array so it matches the size of the chunk.
            end = min(start + rows_pr_iteration, old_size)
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


def create_dataset(old_size: int, new_size: int, old_filename: str, new_filename: str, rows_pr_iteration: int = 20000):
    rows_pr_iteration = min(rows_pr_iteration, old_size)
    new_size = min(new_size, old_size)
    arr = create_randomly_array(old_size=old_size, new_size=new_size)
    # Write the header row to the new files:
    with open(old_filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    colnames.to_csv(new_filename, mode='w')
    # Loop through data in chunks and append to the right dataset:
    start = 0
    with pd.read_csv(old_filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n') as reader:
        for chunk in tqdm(reader, desc='creating dataset', total=int(old_size/rows_pr_iteration), unit='rows encountered in orig. dataset', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            # Get the amount of the split array so it matches the size of the chunk.
            end = min(start + rows_pr_iteration, old_size)
            chunk_split = arr[start:end]
            start += chunk.shape[0]
            # Select the values from the chunk for the dataset:
            rows = chunk[chunk_split == True]
            if rows.shape[0] > 0:
                rows.to_csv(new_filename, mode='a', header=None)


def read_rows(filename: str, idx: int, num: int = 1) -> int:
    return pd.read_csv(filename, encoding='utf-8', lineterminator='\n', skiprows=idx, nrows=num)


def run_split_dataset(size: int, split: Tuple[float, float, float], data_filename: str, train_filename: str, vali_filename: str, test_filename: str, rows_pr_iteration: int = 20000):
    rows = number_of_rows(filename=data_filename,
                          rows_pr_iteration=rows_pr_iteration)
    create_train_vali_and_test_sets(old_size=rows, new_size=size, split=split, data_filename=data_filename, train_filename=train_filename,
                                    vali_filename=vali_filename, test_filename=test_filename, rows_pr_iteration=rows_pr_iteration)


def run_single_dataset(size: int, old_filename: str, new_filename: str, rows_pr_iteration: int = 20000):
    rows = 8528956 #number_of_rows(filename=old_filename,
            #              rows_pr_iteration=rows_pr_iteration)
    create_dataset(old_size=rows, new_size=size, old_filename=old_filename,
                   new_filename=new_filename, rows_pr_iteration=rows_pr_iteration)


# 8528956 rows in original big dataset
if __name__ == '__main__':
    run_single_dataset(size=10000, old_filename="../datasets/sample/news_cleaned_2018_02_13.csv",
                       new_filename='../datasets/sample/dataset.csv', rows_pr_iteration=20000)
    #    vali_filename='../datasets/big/vali.csv', test_filename='../datasets/big/test.csv', rows_pr_iteration=20000)
    # run_split_dataset(size=100000, split=(0.8, 0.1, 0.1), data_filename="../datasets/big/news_cleaned_2018_02_13.csv", train_filename='../datasets/big/train.csv',
    #    vali_filename='../datasets/big/vali.csv', test_filename='../datasets/big/test.csv', rows_pr_iteration=20000)
    # run_split_dataset(size=200, split=(0.8, 0.1, 0.1), data_filename="../datasets/sample/news_sample.csv", train_filename='../datasets/sample/train.csv',
    #   vali_filename='../datasets/sample/vali.csv', test_filename='../datasets/sample/test.csv', rows_pr_iteration=20000)
