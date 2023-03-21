import shutil
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
ROWS_PR_ITERATION = 40000
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


def clean_chunk(chunk: pd.DataFrame, idx_start: int) -> pd.DataFrame:
    # Drop rows with empty content column:
    chunk = chunk.dropna(subset=['content'])
    # Remove rows where type is not one of the specified values:
    chunk = chunk[chunk['type'].isin(TYPES)]
    # Set unique index for each row:
    chunk['id'] = chunk['id'].reset_index(drop=True).index + idx_start
    # Remove index column:
    chunk = chunk.drop(chunk.columns[[0]], axis=1)
    return chunk


def csv_to_h5(csv_filename: str, hdf_filename: str):
    with h5py.File(hdf_filename, 'w') as store:
        with open(csv_filename, encoding='utf-8') as f:
            colnames = next(csv.reader(f))
        # Remove first coloumn (unnamed), which is not used:
        colnames.pop(0)
        cols = len(colnames)
        # Create a dataset:
        data_set = store.create_dataset('data', data=create_empty_string_array(cols), maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))#TODO
        # Set the header row:
        data_set[0] = colnames
        # Read the rest of the rows and assign to dataset:
        original_rows = retained_rows = 0
        for chunk in tqdm(pd.read_csv(csv_filename, encoding='utf-8', dtype=str, chunksize=ROWS_PR_ITERATION, lineterminator='\n'),
                      desc='csv to hdf', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            original_rows += chunk.shape[0]
            chunk = clean_chunk(chunk, retained_rows)
            # Append processed chunk to new file:
            retained_rows += chunk.shape[0]
            data_set.resize((retained_rows+1, cols))
            # Check if chunk is empty. If not, assign to dataset:
            if len(chunk) > 0:
                data_set[-len(chunk):] = chunk.astype(str)
     
            
def h5_to_csv(hdf_filename: str, csv_filename: str):
    with h5py.File(hdf_filename, 'r') as read:
        data = read['data'][0]
        # Convert the header data to a list of strings and save it to CSV
        str_data = [s.decode('utf-8') for s in data]
        pd.DataFrame([str_data]).to_csv(csv_filename, mode='w', header=None, index=False)
        # Loop through the rest of the data and save it to CSV
        for i in range(1, read['data'].shape[0], ROWS_PR_ITERATION):
            # Get the data from the HDF5 file
            data = read['data'][i:i+ROWS_PR_ITERATION]
            # Convert the data to a list of list of strings
            str_data = []
            for d in data:
                str_data.append([s.decode('utf-8') for s in d])
            # Save the data to CSV:
            pd.DataFrame(str_data).to_csv(csv_filename, mode='a', header=None, index=False)


def shuffle_h5(old_filename: str, new_filename: str):
    with h5py.File(old_filename, 'r') as read, h5py.File(new_filename, 'w') as write:
        rows = read['data'].shape[0]-1
        cols = read['data'].shape[1]
        # Create a dataset:
        write_set = write.create_dataset('data', data=create_empty_string_array(cols), maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        # Create a random array of the given size:
        random_arr = create_random_array(size=rows)
        # Set the header row:
        write_set.resize((1, cols))
        write_set[0] = read['data'][0]
        # Loop through the old dataset and take out rows corresponding to randomly created array:
        for i, x in enumerate(tqdm(random_arr, desc='shuffling hdf', unit='rows encountered', colour=TQDM_COLOR), start=1):
            write_set.resize((i+1, cols))
            write_set[i] = read['data'][x]


def run(sample: bool):
    path = "../datasets/sample/" if sample else "../datasets/large/"
    csv_to_h5(csv_filename=path+"raw.csv", hdf_filename=path+"raw.h5")
    shuffle_h5(old_filename=path+"raw.h5", new_filename=path+"shuffled.h5")
    h5_to_csv(hdf_filename=path+"shuffled.h5", csv_filename=path+"dataset.csv")
    

if __name__ == '__main__':
    run(sample=SAMPLE)