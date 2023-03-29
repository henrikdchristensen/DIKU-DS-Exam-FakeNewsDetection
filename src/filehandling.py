import os
import shutil
import csv
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from collections import Counter

TQDM_COLOR = 'magenta'
ROWS_PR_ITERATION = 20000
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


def copy_file(filename: str, new_filename: str):
    print("Copying file...")
    remove_file(new_filename)
    shutil.copyfile(filename, new_filename)


def create_directory(dirname: str):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def remove_directory(dirname: str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def create_random_array(size: int) -> np.ndarray:
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


def tsv_to_csv(file: str, new_file: str, headers: list = None):
    df = pd.read_csv(file, delimiter='\t', header=None)
    if headers is not None:
        df.columns = headers
    df.to_csv(new_file, index=False)


def combine_csv_files(files: list, new_file: str):
    df = pd.concat([pd.read_csv(f) for f in files])
    df.to_csv(new_file, index=False)


def create_empty_string_array(cols: int) -> np.ndarray:
    arr = np.empty((1, cols), dtype=object)
    arr.fill('')
    return arr


def decode_2d(data: np.ndarray) -> list[list]:
    str_data = []
    for d in data:
        str_data.append([s.decode('utf-8') for s in d])
    return str_data


def decode_1d(data: np.ndarray) -> list:
    return [s.decode('utf-8') for s in data]


def decode_dict(data: dict) -> dict:
    return {k.decode('utf-8'): v for k, v in data.items()}


def csv_to_h5(csv_filename: str, h5_filename: str):
    with h5py.File(h5_filename, 'w') as store:
        with open(csv_filename, encoding='utf-8') as f:
            colnames = next(csv.reader(f))
        # Remove first coloumn (unnamed), which is not used:
        colnames.pop(0)
        cols = len(colnames)
        # Create dataset:
        data_set = store.create_dataset('data', data=create_empty_string_array(
            cols), maxshape=(None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
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


def statistics(file: str, output_path: str = None):
    # Initialize counters:
    total_rows = total_cols = 0
    type_counter = Counter()
    # Iterate over all files:
    for chunk in tqdm(pd.read_csv(file, encoding='utf-8', chunksize=ROWS_PR_ITERATION),
                      desc='csv to h5', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
        rows, cols = chunk.shape
        total_rows += rows
        total_cols = cols
        for i in tqdm(range(1, rows, ROWS_PR_ITERATION), desc='creating statistics', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            type_counter.update(chunk[i:i+ROWS_PR_ITERATION, COLS['type']])
    # Add statistics to dataframes:
    total_rows_df = pd.DataFrame([['Number of rows', total_rows]], columns=['Statistic', 'Count'])
    total_cols_df = pd.DataFrame([['Number of cols', total_cols]], columns=['Statistic', 'Count'])
    # Sort dataframes by count and reset index:
    type_df = pd.DataFrame(list(type_counter.items()), columns=['Types', 'Count']).sort_values(
        by='Count', ascending=False).reset_index(drop=True)
    # Print statistics to console without index:
    print(total_rows_df.to_string(index=False))
    print(total_cols_df.to_string(index=False))
    print(type_df.to_string(index=False))
    # Save to file if output_file is specified:
    if output_path is not None:
        total_rows_df.to_csv(output_path+"statistics.csv", mode='w', index=False, header=True)
        total_cols_df.to_csv(output_path+"statistics.csv", mode='a', index=False, header=False)
        type_df.to_csv(output_path+"statistics.csv", mode='a', index=False, header=True)
        print("Statistics added to csv file")


def h5_to_csv(h5_filename: str, csv_filename: str):
    with h5py.File(h5_filename, 'r') as read:
        rows = read['data'].shape[0]
        # Save the header row to CSV
        pd.DataFrame([decode_1d(read['data'][0])]).to_csv(csv_filename, mode='w', header=None, index=False)
        # Loop through the rest of the data and save it to CSV
        for i in tqdm(range(1, rows, ROWS_PR_ITERATION),
                      desc='h5 to csv', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            # Get the data from the HDF5 file
            data = read['data'][i:i+ROWS_PR_ITERATION]
            # Decode the data:
            data = decode_2d(data)
            # Save the data to CSV:
            pd.DataFrame(data).to_csv(csv_filename, mode='a', header=None, index=False)


def shuffle_h5(filename: str, new_filename: str):
    with h5py.File(filename, 'r') as read, h5py.File(new_filename, 'w') as write:
        rows = read['data'].shape[0]-1
        cols = read['data'].shape[1]
        # Create a dataset:
        write_set = write.create_dataset('data', data=create_empty_string_array(
            cols), maxshape=(None, cols), dtype=h5py.string_dtype(encoding='utf-8'))
        # Create a random array of the given size:
        random_arr = create_random_array(size=rows)
        # Set the header row:
        write_set.resize((1, cols))
        write_set[0] = read['data'][0]
        # Loop through the old dataset and take out rows corresponding to randomly created array:
        for i, x in enumerate(tqdm(random_arr, desc='shuffle h5', unit='rows', colour=TQDM_COLOR), start=1):
            write_set.resize((i+1, cols))
            write_set[i] = read['data'][x]


def run():
    choice = input("Press 's' for sample or 'l' for large dataset or 'x' to Exit: ")
    if choice == 'x':
        print("exiting")
        return
    elif choice == 's':
        path = "../datasets/sample/"
    elif choice == 'l':
        path = "../datasets/large/"
    else:
        print("Invalid choice - exiting")
        return
    choice = input("Press 'y' for convert csv to h5 or 'Enter' to continue or 'x' to Exit: ")
    if choice == 'y':
        csv_to_h5(csv_filename=path+"raw.csv", h5_filename=path+"raw.h5")
    elif choice == 'x':
        print("exiting")
        return
    # Copy the raw file to a new file:
    copy_file(filename=path+"raw.h5", new_filename=path+"raw_copy.h5")
    # Get the statistics:
    statistics(path+"raw_copy.h5", output_path=path)
    # Shuffle the data:
    shuffle_h5(filename=path+"raw_copy.h5", new_filename=path+"shuffled.h5")
    # Convert h5 files to csv files:
    h5_to_csv(h5_filename=path+"shuffled.h5", csv_filename=path+"shuffled.csv")


if __name__ == '__main__':
    run()
