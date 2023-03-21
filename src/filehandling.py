from enum import IntEnum
import shutil
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm

TQDM_COLOR = 'magenta'
TYPES_DICT = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 'bias', 
              'satire', 'state', 'reliable', 'clickbait', 'political']
SAMPLE = False
ROWS_PR_ITERATION = 20000

def remove_file(filename: str):
    if os.path.exists(filename):
        os.remove(filename)

def create_directory(dirname: str):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def remove_directory(dirname: str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

def create_random_array(size:int) -> np.ndarray:
    # Create a numpy array of the given size and set values from 0 to size-1:
    arr = np.arange(size)
    # Shuffle the indexes of the array:
    np.random.shuffle(arr)
    return arr

def csv_split(filename: str, dirname: str = 'csv-chunks', rows_pr_iteration: int = 20000, padding: int = 4):
    # Get and set header row:
    with open(filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    # Remove existing directory and create new:
    remove_directory(dirname)
    create_directory(dirname)
    # Add each chunk to a file in the directory:
    for i, c in tqdm(enumerate(pd.read_csv(filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n')),
                     desc='csv split', unit='splits', colour=TQDM_COLOR):
        pd.concat([colnames, c], ignore_index=True).to_csv(
            f'{dirname}/{i+1:0{padding}}.csv', index=False)

def remove_unwanted(filename: str, new_filename: str, rows_pr_iteration: int = 20000) -> int:
    with open(filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    colnames.to_csv(new_filename, mode='w', index=False)
    original_rows = 0
    retained_rows = 0
    with pd.read_csv(filename, encoding='utf-8', chunksize=rows_pr_iteration, lineterminator='\n') as reader:
        for chunk in tqdm(reader, desc='remove missing/incorrect values', unit='rows', unit_scale=rows_pr_iteration, colour=TQDM_COLOR):
            original_rows += chunk.shape[0]
            # Drop rows with empty content column:
            chunk = chunk.dropna(subset=['content'])
            # Remove rows where type is not one of the specified values:
            chunk = chunk[chunk['type'].isin(TYPES_DICT)]
            # Set unique index for each row:
            chunk.iloc[:,0] = chunk.iloc[:,0].reset_index(drop=True).index + retained_rows
            chunk.to_csv(new_filename, mode='a', header=None, index=False)
            retained_rows += chunk.shape[0]
    print(f"Removed rows: {original_rows-retained_rows}\n(rows before: {original_rows}, rows after (retained): {retained_rows})")
    return retained_rows

def read_rows(filename: str, idx: int, num: int = 1) -> int:
    return pd.read_csv(filename, encoding='utf-8', lineterminator='\n', skiprows=idx, nrows=num)

def create_dataset(size:int, old_filename: str, new_filename: str):
    random_arr = create_random_array(size=size)
    # Write the header row to the new files:
    with open(old_filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    colnames.to_csv(new_filename, mode='w', index=False)
    # Loop through cleaned dataset and take out rows corresponding to randomly created array:
    for index in tqdm(random_arr, desc='creating dataset', unit='rows', colour=TQDM_COLOR):
        read_rows(old_filename, index, 1).to_csv(new_filename, mode='a', header=None, index=False)

def run(sample: bool, rows_pr_iteration: int = ROWS_PR_ITERATION):
    path = "../datasets/sample/" if sample else "../datasets/large/"
    size = remove_unwanted(filename=path+"raw.csv", new_filename=path+"cleaned.csv", rows_pr_iteration=rows_pr_iteration)
    create_dataset(size=size, old_filename=path+"cleaned.csv", new_filename=path+"dataset.csv")

if __name__ == '__main__':
    run(sample=SAMPLE, rows_pr_iteration=ROWS_PR_ITERATION)
