import shutil
from typing import Tuple
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm
import h5py

TQDM_COLOR = 'magenta'
TYPES = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 'bias', 
              'satire', 'state', 'reliable', 'clickbait', 'political']
SAMPLE = False
ROWS_PR_ITERATION = 20000
FILE_SIZE = 10000
PADDING = 3

# TODO: Add h5py to requirements.txt


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
    arr = np.arange(1, size+1)
    # Shuffle the indexes of the array:
    np.random.shuffle(arr)
    return arr


def csv_split(filename: str, dirname: str):
    # Get and set header row:
    with open(filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    # Remove existing directory and create new:
    remove_directory(dirname)
    create_directory(dirname)
    # Create a file for each chunk in the directory:
    for i, c in tqdm(enumerate(pd.read_csv(filename, encoding='utf-8', chunksize=FILE_SIZE, lineterminator='\n')),
                     desc='csv splitting', unit='splits', colour=TQDM_COLOR):
        pd.concat([colnames, c], ignore_index=True).to_csv(f'{dirname}/{i+1:0{PADDING}}.csv', index=False)


def create_empty_string_array(cols: int) -> np.ndarray:
    arr = np.zeros((1, cols), dtype=object)
    arr[0] = ["" for x in range(cols)]
    return arr


def clean_chunk(chunk: pd.DataFrame, idx_start: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dropped_rows = chunk.copy()
    # Remove rows where type is not one of the specified values:
    chunk = chunk[chunk['type'].isin(TYPES)]
    # Save the dropped rows in a separate dataframe:
    dropped_rows = dropped_rows[~dropped_rows['type'].isin(TYPES)]
    # Save the dropped rows with empty content:
    dropped_rows = pd.concat([dropped_rows, chunk[chunk['content'].isna()]])
    # Drop rows with empty content column:
    chunk = chunk.dropna(subset=['content'])
    # Set unique index for each row:
    chunk['id'] = chunk['id'].reset_index(drop=True).index + idx_start
    # Remove index column:
    chunk = chunk.drop(chunk.columns[[0]], axis=1)
    dropped_rows = dropped_rows.drop(dropped_rows.columns[[0]], axis=1)
    return chunk, dropped_rows


def decode_h5_dataset(data):
    str_data = []
    for d in data:
        str_data.append([s.decode('utf-8') for s in d])
    return str_data


def csv_to_h5(csv_filename: str, hdf_filename: str):
    with h5py.File(hdf_filename, 'w') as store, h5py.File(hdf_filename[:-3]+'_dropped.h5', 'w') as dropped_store:
        with open(csv_filename, encoding='utf-8') as f:
            colnames = next(csv.reader(f))
        # Remove first coloumn (unnamed), which is not used:
        colnames.pop(0)
        cols = len(colnames)
        # Create dataset:
        data_set = store.create_dataset('data', data=create_empty_string_array(cols), maxshape=(None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        dropped_set = dropped_store.create_dataset('data', data=create_empty_string_array(cols), maxshape=(None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        # Set the header row:
        data_set[0] = colnames
        dropped_set[0] = colnames
        # Read the rest of the rows and assign to dataset:
        original_rows = retained_rows = dropped_rows = 0
        for chunk in tqdm(pd.read_csv(csv_filename, encoding='utf-8', dtype=str, chunksize=ROWS_PR_ITERATION, lineterminator='\n'),
                      desc='csv to h5', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            original_rows += chunk.shape[0]
            retained, dropped = clean_chunk(chunk, retained_rows)
            # Append processed data to files:
            retained_rows += retained.shape[0]
            dropped_rows += dropped.shape[0]
            data_set.resize((retained_rows+1, cols))
            dropped_set.resize((dropped_rows+1, cols))
            # Check if data is empty - if not, append to dataset:
            if len(retained) > 0:
                data_set[-len(retained):] = retained.astype(str)
            if len(dropped) > 0:
                dropped_set[-len(dropped):] = dropped.astype(str)
        print(f"Original rows: {original_rows}, retained rows: {retained_rows}, dropped rows: {dropped_rows}")
     
            
def h5_to_csv(hdf_filename: str, csv_filename: str):
    with h5py.File(hdf_filename, 'r') as read:
        # Save the header row to CSV
        header = read['data'][0]
        header = [s.decode('utf-8') for s in header]
        pd.DataFrame([header]).to_csv(csv_filename, mode='w', header=None, index=False)
        # Loop through the rest of the data and save it to CSV
        for i in tqdm(range(1, read['data'].shape[0], ROWS_PR_ITERATION),
                      desc='h5 to csv', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            # Get the data from the HDF5 file
            data = read['data'][i:i+ROWS_PR_ITERATION]
            # Save the data to CSV:
            pd.DataFrame(decode_h5_dataset(data)).to_csv(csv_filename, mode='a', header=None, index=False)


def shuffle_h5(old_filename: str, new_filename: str):
    with h5py.File(old_filename, 'r') as read, h5py.File(new_filename, 'w') as write:
        rows = read['data'].shape[0]-1
        cols = read['data'].shape[1]
        # Create a dataset:
        write_set = write.create_dataset('data', data=create_empty_string_array(cols), maxshape=(None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        # Create a random array of the given size:
        random_arr = create_random_array(size=rows)
        # Set the header row:
        write_set.resize((1, cols))
        write_set[0] = read['data'][0]
        # Loop through the old dataset and take out rows corresponding to randomly created array:
        for i, x in enumerate(tqdm(random_arr, desc='shuffle h5', unit='rows', colour=TQDM_COLOR), start=1):
            write_set.resize((i+1, cols))
            write_set[i] = read['data'][x]


def run(sample: bool):
    path = "../datasets/sample/" if sample else "../datasets/large/"
    csv_to_h5(csv_filename=path+"raw.csv", hdf_filename=path+"raw.h5")
    shuffle_h5(old_filename=path+"raw.h5", new_filename=path+"shuffled.h5")
    h5_to_csv(hdf_filename=path+"shuffled.h5", csv_filename=path+"dataset.csv")
    

if __name__ == '__main__':
    run(sample=SAMPLE)