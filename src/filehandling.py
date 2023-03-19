from collections import Counter
from typing import Dict, Tuple
from enum import IntEnum
import shutil
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm

TQDM_COLOR = 'magenta'
types = {'fake': 1, 'conspiracy': 2, 'junksci': 3, 'hate': 4, 'unreliable': 5,
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


types = {'fake': 1, 'conspiracy': 2, 'junksci': 3, 'hate': 4, 'unreliable': 5,
         'bias': 6, 'satire': 7, 'state': 8, 'reliable': 9, 'clickbait': 10, 'political': 11}


def create_randomly_balanced_array(types_list: list, new_size: int, threshold: float = 0.05) -> np.ndarray:
    # Calculate the counts of each type in the original array
    type_counts = np.bincount(types_list, minlength=len(types))
    print(type_counts)
    # Calculate the target count for each type in the new array
    target_counts = np.around(
        type_counts.sum() * np.array(list(types.values())) / sum(types.values()))
    # Calculate the maximum allowed deviation from the target count, based on the threshold
    max_deviation = np.around(threshold * target_counts.sum())
    # Generate the new array
    new_array = np.zeros(len(types), dtype=bool)
    i = 0
    while i < new_size:
        # Select a random type index to add to the new array
        type_index = np.random.randint(len(types))
        # Check if adding this type would exceed the maximum deviation from the target count
        if abs(target_counts[type_index] - np.count_nonzero(new_array == type_index)) > max_deviation:
            continue
        # Add the type to the new array
        new_array[type_index] = True
        i += 1
    # Shuffle the indexes of the new array and return
    new_array = np.random.permutation(new_array)
    return new_array


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


def data_preparation(filename: str, new_filename: str, rows_pr_iteration: int = 20000) -> Tuple[int, np.ndarray]:
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
            chunk = chunk[chunk['type'].isin(types)]
            retained_rows += chunk.shape[0]
            chunk.to_csv(new_filename, mode='a', header=None)
            # Append the 'type' column to the type_array
            # Append the value of the 'type' column to the type_array
            type_arr = np.append(type_arr, chunk['type'].map(types))
            #type_arr.extend([types[t] for t in chunk['type'].tolist()])
    print(
        f"Removed rows: {original_rows-retained_rows}\n(original rows: {original_rows}, retained rows: {retained_rows})")
    print("\nType counts:")
    unique, counts = np.unique(type_arr, return_counts=True)
    for i in range(len(unique)):
        print(
            f"{list(types.keys())[list(types.values()).index(unique[i])]}: {counts[i]}")
    missing_types = set(types.values()) - set(unique)
    if len(missing_types) > 0:
        print("\nMissing types:")
        for t in missing_types:
            print(
                f"{list(types.keys())[list(types.values()).index(t)]}")
    return (retained_rows, type_arr)


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
        for chunk in tqdm(reader, desc='splitting data into: train-, vali-, and test set', total=int(old_size/rows_pr_iteration), unit='rows encountered', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            # Get the amount of the split array so it matches the size of the chunk.
            end = min(start + rows_pr_iteration, old_size)
            chunk_split = split[start:end]
            start += chunk.shape[0]
            # Select the values from the chunk for the train- or vali- or test dataset
            # from the chunk if it matches the shuffled split array:
            train_rows = chunk[chunk_split is Set.TRAIN]
            if train_rows.shape[0] > 0:
                train_rows.to_csv(train_filename, mode='a', header=None)
            vali_rows = chunk[chunk_split is Set.VALI]
            if vali_rows.shape[0] > 0:
                vali_rows.to_csv(vali_filename, mode='a', header=None)
            test_rows = chunk[chunk_split is Set.TEST]
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
        for chunk in tqdm(reader, desc='creating dataset', total=int(old_size/rows_pr_iteration), unit='rows encountered', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
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


ROWS_PR_ITERATION = 20000
CLEANED_ROWS_LARGE = 7273069  # counted already
CLEANED_ROWS_SAMPLE = 232  # counted already
NEW_SIZE_LARGE = 100000
NEW_SIZE_SAMPLE = 200


def run(sample: bool = True, rows_cleaned: bool = False, rows_pr_iteration: int = ROWS_PR_ITERATION, split=False, new_size: int = NEW_SIZE_SAMPLE):
    if sample:
        old_size = CLEANED_ROWS_SAMPLE
        new_size = min(new_size, NEW_SIZE_SAMPLE)
        path = "../datasets/sample/"
    else:
        old_size = CLEANED_ROWS_LARGE
        new_size = min(new_size, NEW_SIZE_LARGE)
        path = "../datasets/large/"
    create_directory(path)
    if not rows_cleaned:
        (old_size, types) = data_preparation(filename=path+"raw.csv",
                                             new_filename=path+"cleaned.csv", rows_pr_iteration=rows_pr_iteration)
    if not split:
        create_dataset(old_size=old_size, new_size=new_size, old_filename=path+"cleaned.csv",
                       new_filename=path+"dataset.csv", rows_pr_iteration=rows_pr_iteration)
    else:
        create_train_vali_and_test_sets(old_size=old_size, new_size=new_size, split=(0.8, 0.1, 0.1), data_filename=path+"cleaned.csv", train_filename=path+"train.csv",
                                        vali_filename=path+"vali.csv", test_filename=path+"test.csv", rows_pr_iteration=rows_pr_iteration)


if __name__ == '__main__':
    # run(sample=True, rows_cleaned=False,
    #     rows_pr_iteration=ROWS_PR_ITERATION, split=False, new_size=200)
    data_preparation(filename="../datasets/sample/raw.csv",
                     new_filename="../datasets/sample/cleaned.csv", rows_pr_iteration=ROWS_PR_ITERATION)
