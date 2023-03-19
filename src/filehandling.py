from collections import Counter
from typing import Dict, Tuple
from enum import IntEnum
import shutil
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm
import warnings

TQDM_COLOR = 'magenta'
types_dict = {'fake': 1, 'conspiracy': 2, 'junksci': 3, 'hate': 4, 'unreliable': 5,
              'bias': 6, 'satire': 7, 'state': 8, 'reliable': 9, 'clickbait': 10, 'political': 11}


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


def create_random_array(balance_classes: bool, all_classes_must_exist: bool, balance_hard: bool, types: np.ndarray, new_size: int) -> np.ndarray:
    print("\nType counts:")
    unique, counts = np.unique(types, return_counts=True)
    for i in range(len(unique)):
        print(
            f"{list(types_dict.keys())[list(types_dict.values()).index(unique[i])]}: {counts[i]}")
    missing_types = set(types_dict.values()) - set(unique)
    if len(missing_types) > 0:
        print("\nMissing types:")
        for t in missing_types:
            print(
                f"{list(types_dict.keys())[list(types_dict.values()).index(t)]}")
        if all_classes_must_exist:
            # Raise error if hard_balance is True and there are missing types
            raise Exception("All classes doesn't exists.")
    if not balance_classes:
        # Create a numpy array of the given size and set all to zeroes
        arr = np.zeros(len(types), dtype=bool)
        # Set the values for the three splits
        # index 0 (inclusive) up to index "split1" (exclusive)
        arr[0:new_size] = True
        # Shuffle the indexes of the array and return
        np.random.shuffle(arr)
    else:
        # Create a dictionary to keep track of the number of elements of each type
        counts = {t: np.count_nonzero(types == t)
                  for t in np.unique(types)}
        # Calculate the number of elements to include for each type
        if balance_hard:
            min_count = min(counts.values())
            type_size = min_count
        else:
            type_size = round(new_size / len(counts))
        print(f"Number of elements to take for each type: {type_size}")
        if new_size % len(counts) != 0:
            warnings.warn(
                f"\nWarning! The new_size={new_size} is not divisible by the number of {len(counts)} types.\nThe final array will contain True values less than the given new_size.\n")
        # Create a numpy array of the given size and set all to False
        arr = np.zeros(len(types), dtype=bool)
        # Iterate over the types and set the corresponding indexes to True
        for key, value in counts.items():
            indices = [i for i, x in enumerate(types) if x == key]
            num_elements = len(indices)
            if num_elements < type_size:
                arr[indices] = True
                warnings.warn(
                    f"Warning! type {key} has only {num_elements} elements - adding all.\nThe final array will contain True values less than the given new_size.\n")
            else:
                arr[np.random.choice(indices, type_size, replace=False)] = True
        print(f"Number of True value in the final array: {np.sum(arr)}")
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


def data_preparation(filename: str, new_filename: str, rows_pr_iteration: int = 20000) -> np.ndarray:
    with open(filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    colnames.to_csv(new_filename, mode='w')
    original_rows = 0
    retained_rows = 0
    type_arr = np.empty(0, dtype=int)
    with pd.read_csv(filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n') as reader:
        for chunk in tqdm(reader, desc='remove missing/false values', unit='rows encountered', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            original_rows += chunk.shape[0]
            # Drop rows with empty content column:
            chunk = chunk.dropna(subset=['content'])
            # Remove rows where type is not one of the specified values:
            chunk = chunk[chunk['type'].isin(types_dict)]
            retained_rows += chunk.shape[0]
            chunk.to_csv(new_filename, mode='a', header=None)
            # Append the 'type' column to the type_array
            # Append the value of the 'type' column to the type_array
            type_arr = np.append(
                type_arr, chunk['type'].map(types_dict))
    print(
        f"Removed rows: {original_rows-retained_rows}\n(original rows: {original_rows}, retained rows: {retained_rows})")
    return type_arr


def create_dataset(types: np.ndarray, new_size: int, old_filename: str, new_filename: str, rows_pr_iteration: int = 20000, balance_classes: bool = False, all_classes_must_exist: bool = False, balance_hard: bool = False):
    old_size = len(types)
    rows_pr_iteration = min(rows_pr_iteration, old_size)
    new_size = min(new_size, old_size)
    arr = create_random_array(
        balance_classes=balance_classes, all_classes_must_exist=all_classes_must_exist, balance_hard=balance_hard, types=types, new_size=new_size)
    # Write the header row to the new files:
    with open(old_filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    colnames.to_csv(new_filename, mode='w')
    # Loop through data in chunks and append to the right dataset:
    start = 0
    with pd.read_csv(old_filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n') as reader:
        for chunk in tqdm(reader, desc='creating dataset', total=int(old_size/rows_pr_iteration), unit='rows encountered', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            # Get the amount of the split array so it matches the size of the chunk.
            end = min(start + rows_pr_iteration, old_size)
            chunk_split = arr[start:end]
            start += chunk.shape[0]
            # Select the values from the chunk for the dataset:
            rows = chunk[chunk_split == True]
            if rows.shape[0] > 0:
                rows.to_csv(new_filename, mode='a', header=None)
    print("Dataset created")


def read_rows(filename: str, idx: int, num: int = 1) -> int:
    return pd.read_csv(filename, encoding='utf-8', lineterminator='\n', skiprows=idx, nrows=num)


#CLEANED_ROWS_LARGE = 7273069
#CLEANED_ROWS_SAMPLE = 232
ROWS_PR_ITERATION = 20000


def run(sample: bool = True, rows_pr_iteration: int = ROWS_PR_ITERATION, new_size: int = 0, balance_classes: bool = False, all_classes_must_exist: bool = False, balance_hard: bool = False):
    if sample:
        path = "../datasets/sample/"
    else:
        path = "../datasets/large/"
    create_directory(path)
    types = data_preparation(filename=path+"raw.csv",
                             new_filename=path+"cleaned.csv", rows_pr_iteration=rows_pr_iteration)
    create_dataset(types=types, new_size=new_size, old_filename=path+"cleaned.csv",
                   new_filename=path+"dataset.csv", rows_pr_iteration=rows_pr_iteration, balance_classes=balance_classes, all_classes_must_exist=all_classes_must_exist, balance_hard=balance_hard)


if __name__ == '__main__':
    run(sample=False, rows_cleaned=False,
        rows_pr_iteration=ROWS_PR_ITERATION, new_size=100000, balance_classes=True, all_classes_must_exist=False, balance_hard=False)
