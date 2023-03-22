import os
import shutil
import csv
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from collections import Counter

TQDM_COLOR = 'magenta'
TYPES = [b'fake', b'conspiracy', b'junksci', b'hate', b'unreliable', b'bias', 
            b'satire', b'state', b'reliable', b'clickbait', b'political']
SAMPLE = True
ROWS_PR_ITERATION = 20
FILE_SIZE = 10000
PADDING = 3
COLS = {
    'id': 0,
    'domain': 1,
    'type': 2,
    'url': 3,
    'content': 4,
    'scraped_at': 5,
    'inserted_at': 6,
    'updated_at': 7,
    'title': 8,
    'authors': 9,
    'keywords': 10,
    'meta_keywords': 11,
    'meta_description': 12,
    'tags': 13,
    'summary': 14
}


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


def decode_2d_array(data: np.ndarray) -> np.ndarray:
    str_data = []
    for d in data:
        str_data.append([s.decode('utf-8') for s in d])
    return str_data


def decode_1d_array(data: np.ndarray) -> np.ndarray:
    return [s.decode('utf-8') for s in data]


def csv_to_h5(csv_filename: str, h5_filename: str):
    with h5py.File(h5_filename, 'w') as store:
        with open(csv_filename, encoding='utf-8') as f:
            colnames = next(csv.reader(f))
        # Remove first coloumn (unnamed), which is not used:
        colnames.pop(0)
        cols = len(colnames)
        # Create dataset:
        data_set = store.create_dataset('data', data=create_empty_string_array(cols), maxshape=(None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        # Set the header row:
        data_set[0] = colnames
        # Read the rest of the rows and assign to dataset:
        rows = 0
        for chunk in tqdm(pd.read_csv(csv_filename, encoding='utf-8', dtype=str, chunksize=ROWS_PR_ITERATION, lineterminator='\n'),
                      desc='csv to h5', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            # Remove first column (unnamed), which is not used:
            chunk = chunk.drop(chunk.columns[[0]], axis=1)
            # Set unique index for each row in 'id' column and remove index column:
            chunk[chunk.columns[COLS['id']]] = range(rows, rows+chunk.shape[0])
            # Resizing dataset:
            rows += chunk.shape[0]
            data_set.resize((rows+1, cols))
            # Check if data is empty - if not, append to dataset:
            if len(chunk) > 0:
                data_set[-len(chunk):] = chunk.astype(str)

    
def remove_unwanted_rows(data_filename: str, retained_filename: str, removed_filename: str):
    with h5py.File(data_filename, 'r') as data_store, h5py.File(retained_filename, 'w') as retained_store, h5py.File(removed_filename, 'w') as removed_store:
        # Create dataset:
        header = data_store['data'][0]
        rows, cols = data_store['data'].shape
        retained_set = retained_store.create_dataset('data', data=create_empty_string_array(cols), maxshape=(None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        removed_set = removed_store.create_dataset('data', data=create_empty_string_array(cols), maxshape=(None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        retained_set[0] = header
        removed_set[0] = header
        # Initialize counters:
        retained_rows = faulty_content_rows = faulty_type_rows = 0
        for i in tqdm(range(1, rows, ROWS_PR_ITERATION), desc=f'remove unwanted rows', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            chunk = data_store['data'][i:i+ROWS_PR_ITERATION]
            decoded = decode_1d_array(chunk[:, COLS['content']])
            # Remove rows with empty content and incorrect/missing types:
            mask_content = np.logical_and(chunk[:, COLS['content']] != b'nan', ~np.char.startswith(decoded, 'ERROR'))
            mask_type = np.isin(chunk[:, COLS['type']], TYPES)
            mask = np.logical_and(mask_content, mask_type)
            retained_chunk = chunk[mask]
            removed_chunk = chunk[~mask]
            # Update counters:
            retained_rows += np.sum(mask)
            faulty_content_rows += np.sum(~mask_content)
            faulty_type_rows += np.sum(~mask_type)
            if retained_chunk.shape[0] > 0:
                retained_set.resize(retained_set.shape[0] + retained_chunk.shape[0], axis=0)
                retained_set[-retained_chunk.shape[0]:] = retained_chunk
            if removed_chunk.shape[0] > 0:
                removed_set.resize(removed_set.shape[0] + removed_chunk.shape[0], axis=0)
                removed_set[-removed_chunk.shape[0]:] = removed_chunk
    print(f"Original rows: {rows-1}, retained rows: {retained_rows}, rows with faulty content: {faulty_content_rows}, rows with faulty type: {faulty_type_rows}")


def statistics(*h5_filenames: str):
    # Initialize counters:
    total_rows = 0
    total_cols = 0
    domains_counter = Counter()
    types_counter = Counter()
    # Iterate over all files:
    for h5_filename in h5_filenames:
        with h5py.File(h5_filename, 'r') as data_store:
            rows, cols = data_store['data'].shape
            total_rows += rows - 1
            total_cols = cols
            for i in tqdm(range(1, rows, ROWS_PR_ITERATION), desc=f'processing {h5_filename}', unit='rows', colour=TQDM_COLOR):
                chunk = data_store['data'][i:i+ROWS_PR_ITERATION]
                # Decode data:
                domains = decode_1d_array(chunk[:, COLS['domain']])
                types = decode_1d_array(chunk[:, COLS['type']])
                # Update counters:
                domains_counter.update(domains)
                types_counter.update(types)
    # Print statistics:
    print(f"Number of rows: {total_rows}, number of columns: {total_cols}")
    print(f"Number of different domains: {dict(domains_counter)}")
    print(f"Number of different types: {dict(types_counter)}")

     
def h5_to_csv(h5_filename: str, csv_filename: str):
    with h5py.File(h5_filename, 'r') as read:
        rows = read['data'].shape[0]
        # Save the header row to CSV
        pd.DataFrame([decode_1d_array(read['data'][0])]).to_csv(csv_filename, mode='w', header=None, index=False)
        # Loop through the rest of the data and save it to CSV
        for i in tqdm(range(1, rows, ROWS_PR_ITERATION),
                      desc='h5 to csv', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            # Get the data from the HDF5 file
            data = read['data'][i:i+ROWS_PR_ITERATION]
            # Decode the data:
            data = decode_2d_array(data)
            # Save the data to CSV:
            pd.DataFrame(data).to_csv(csv_filename, mode='a', header=None, index=False)


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
    csv_to_h5(csv_filename=path+"raw.csv", h5_filename=path+"raw.h5")
    statistics(path+"raw.h5")
    remove_unwanted_rows(data_filename=path+"raw.h5", retained_filename=path+"retained.h5", removed_filename=path+"removed.h5")
    shuffle_h5(old_filename=path+"retained.h5", new_filename=path+"retained_shuffled.h5")
    h5_to_csv(h5_filename=path+"retained.h5", csv_filename=path+"retained.csv")
    h5_to_csv(h5_filename=path+"retained_shuffled.h5", csv_filename=path+"retained_shuffled.csv")
    h5_to_csv(h5_filename=path+"removed.h5", csv_filename=path+"removed.csv")

if __name__ == '__main__':
    run(sample=SAMPLE)