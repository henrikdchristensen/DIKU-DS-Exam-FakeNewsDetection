import shutil
import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm
import math
import h5py

TQDM_COLOR = 'magenta'
TYPES = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 'bias', 
              'satire', 'state', 'reliable', 'clickbait', 'political']
SAMPLE = True
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
    arr = np.arange(size)
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

def remove_unwanted(filename: str, new_filename: str) -> int:
    # Write the header row to the new file:
    with open(filename, encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    # Remove first coloumn (unnamed), which is not used:
    colnames.drop('', inplace=True, axis=1)
    colnames.to_csv(new_filename, mode='w', index=False)
    original_rows = retained_rows = 0
    with pd.read_csv(filename, encoding='utf-8', chunksize=ROWS_PR_ITERATION, lineterminator='\n') as reader:
        for chunk in tqdm(reader, desc='remove unwanted rows', unit='rows encountered', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            original_rows += chunk.shape[0]
            # Drop rows with empty content column:
            chunk = chunk.dropna(subset=['content'])
            # Remove rows where type is not one of the specified values:
            chunk = chunk[chunk['type'].isin(TYPES)]
            # Set unique index for each row:
            chunk['id'] = chunk['id'].reset_index(drop=True).index + retained_rows
            # Remove index column:
            chunk = chunk.reset_index(drop=True)
            # Append processed chunk to new file:
            chunk.to_csv(new_filename, mode='a', header=None, index=False)
            retained_rows += chunk.shape[0]
    print(f"Removed rows: {original_rows-retained_rows}\n(rows before: {original_rows}, rows after (retained): {retained_rows})")
    return retained_rows

def read_rows(filename: str, idx: int, num: int = 1) -> int:
    return pd.read_csv(filename, encoding='utf-8', lineterminator='\n', skiprows=idx, nrows=num)

def create_dataset(size:int, old_filename: str, new_filename: str):
    temp_dir = "temp"
    # Split the old file into chunks for faster processing:
    csv_split(filename=old_filename, dirname=temp_dir, file_size=FILE_SIZE, padding=PADDING)
    # Create a random array of the given size:
    random_arr = create_random_array(size=size)
    # Write the header row to the new files:
    with open(f'{temp_dir}/{1:0{PADDING}}.csv', encoding='utf-8') as f:
        colnames = pd.DataFrame(columns=next(csv.reader(f)))
    colnames.to_csv(new_filename, mode='w', index=False)
    # Loop through cleaned dataset and take out rows corresponding to randomly created array:
    for i in tqdm(random_arr, desc='creating dataset', unit='rows encountered', colour=TQDM_COLOR):
        # Find the right chunk file:
        file_num = math.ceil((i+1)/FILE_SIZE)
        # Find the row index of the row to be read:
        row_idx = i-(file_num-1)*FILE_SIZE
        # Read the row and append it to the new file:
        read_rows(f'{temp_dir}/{file_num:0{PADDING}}.csv', row_idx, 1).to_csv(new_filename, mode='a', header=None, index=False)
    # Remove the temporary directory:
    #remove_directory(temp_dir)#TODO
    
def create_empty_string_array(cols: int) -> np.ndarray:
    arr = np.zeros((1, cols), dtype=object)
    arr[0] = ["" for x in range(cols)]
    return arr

def csv_to_hdf(csv_filename: str, hdf_filename: str):
    with h5py.File(hdf_filename, 'w') as store:
        with open(csv_filename, encoding='utf-8') as f:
            colnames = next(csv.reader(f))
        colnames.pop(0)
        cols = len(colnames)
        data_set = store.create_dataset('data', data=create_empty_string_array(cols), maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))#TODO
        # Set the header row:
        data_set[0] = colnames
        # Read the rest of the rows and assign to dataset:
        original_rows = retained_rows = 0
        for chunk in tqdm(pd.read_csv(csv_filename, encoding='utf-8', dtype=str, chunksize=ROWS_PR_ITERATION, lineterminator='\n'),
                      desc='csv to hdf', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            original_rows += chunk.shape[0]
            # Drop rows with empty content column:
            chunk = chunk.dropna(subset=['content'])
            # Remove rows where type is not one of the specified values:
            chunk = chunk[chunk['type'].isin(TYPES)]
            # Set unique index for each row:
            chunk['id'] = chunk['id'].reset_index(drop=True).index + retained_rows
            # Remove index column:
            chunk = chunk.reset_index(drop=True)
            chunk = chunk.drop(chunk.columns[[0]], axis=1) 
            # Append processed chunk to new file:
            retained_rows += chunk.shape[0]
            data_set.resize((retained_rows+1, cols))
            data_set[-len(chunk):] = chunk.astype(str)
        return retained_rows, cols

def shuffle_hdf(old_filename: str, new_filename: str, rows: int, cols: int):
    with h5py.File(old_filename, 'r') as read , h5py.File(new_filename, 'w') as write:
        write_set = write.create_dataset('data', data=create_empty_string_array(cols), maxshape=(
            None, cols), dtype=h5py.string_dtype(encoding='utf-8'))#TODO: cols and maxshape
        random_arr = create_random_array(size=rows)
        write_set.resize((1, cols))
        write_set[0] = read['data'][0]
        for i, j in enumerate(tqdm(random_arr, desc='shuffling hdf', unit='rows encountered', colour=TQDM_COLOR)):
            write_set.resize((i+2, cols))
            write_set[i+1] = read['data'][j]
            
def hdf_to_csv(hdf_filename: str, csv_filename: str):
    with h5py.File(hdf_filename, 'r') as read:
        data = read['data'][0]
        # Convert the data to a list of strings
        str_data = [s.decode('utf-8') for s in data]
        pd.DataFrame([str_data]).to_csv(csv_filename, mode='w', header=None, index=False)
        for i in range(read['data'].shape[0]-1):
            # Get the data from the HDF5 file
            data = read['data'][i+1]
            # Convert the data to a list of strings
            str_data = [s.decode('utf-8') for s in data]
            # Convert the list of strings to a DataFrame and save it to CSV
            pd.DataFrame([str_data]).to_csv(csv_filename, mode='a', header=None, index=False)
    

def run(sample: bool):
    path = "../datasets/sample/" if sample else "../datasets/large/"
    rows, cols = csv_to_hdf(csv_filename=path+"raw.csv", hdf_filename=path+"raw.h5")
    shuffle_hdf(old_filename=path+"raw.h5", new_filename=path+"shuffled.h5", rows=rows, cols=cols)
    hdf_to_csv(hdf_filename=path+"shuffled.h5", csv_filename=path+"dataset.csv")
    

if __name__ == '__main__':
    run(sample=SAMPLE)