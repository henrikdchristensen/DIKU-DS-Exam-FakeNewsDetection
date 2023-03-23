import os
import shutil
import csv
from typing import Tuple
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from collections import Counter

TQDM_COLOR = 'magenta'
SAMPLE = True
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
WORDS_CONTENT = 5
CLEAN_COUNT = 10
TYPES = [b'fake', b'conspiracy', b'junksci', b'hate', b'unreliable', b'bias', 
         b'satire', b'state', b'reliable', b'clickbait', b'political']

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
        for i in tqdm(range(1, rows, ROWS_PR_ITERATION), desc='remove unwanted rows', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
            chunk = data_store['data'][i:i+ROWS_PR_ITERATION]
            decoded = decode_1d(chunk[:, COLS['content']])
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


def statistics(*h5_filenames: str, output_file: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Initialize counters:
    total_rows = 0
    total_cols = 0
    content_start_counter = Counter()
    content_end_counter = Counter()
    domain_counter = Counter()
    type_counter = Counter()
    # Iterate over all files:
    for h5_filename in h5_filenames:
        with h5py.File(h5_filename, 'r') as data_store:
            rows, cols = data_store['data'].shape
            total_rows += rows - 1
            total_cols = cols
            for i in tqdm(range(1, rows, ROWS_PR_ITERATION), desc='creating statistics', unit='rows', unit_scale=ROWS_PR_ITERATION, colour=TQDM_COLOR):
                chunk = data_store['data'][i:i+ROWS_PR_ITERATION]
                # Update counters:
                content_start = [c.decode('utf-8').split(' ')[:WORDS_CONTENT] for c in chunk[:, COLS['content']]]
                content_start = [' '.join(c) for c in content_start]
                content_start_counter.update(content_start)
                content_end = [c.decode('utf-8').split(' ')[:WORDS_CONTENT] for c in chunk[:, COLS['content']]]
                content_end = [' '.join(c) for c in content_end]
                content_end_counter.update(content_end)
                domain_counter.update(chunk[:, COLS['domain']])
                type_counter.update(chunk[:, COLS['type']])
    domain_counter = decode_dict(domain_counter)
    type_counter = decode_dict(type_counter)
    # Add statistics to dataframes:
    total_rows_df = pd.DataFrame([['Number of rows', total_rows]], columns=['Statistic', 'Count'])
    total_cols_df = pd.DataFrame([['Number of cols', total_cols]], columns=['Statistic', 'Count'])
    type_df = pd.DataFrame(list(type_counter.items()), columns = ['Types', 'Count']).sort_values(by='Count', ascending=False).reset_index(drop=True)
    domain_df = pd.DataFrame(list(domain_counter.items()), columns = ['Domain', 'Count']).sort_values(by='Count', ascending=False).reset_index(drop=True)
    content_start_df = pd.DataFrame(list(content_start_counter.items()), columns = ['Content start', 'Count'])
    content_end_df = pd.DataFrame(list(content_end_counter.items()), columns = ['Content end', 'Count'])
    # Filter DataFrame to include only counts greater than 10:
    content_start_df = content_start_df[content_start_df['Count'] > 5]
    content_end_df = content_end_df[content_end_df['Count'] > 5]
    # Sort DataFrame by count in descending order:
    content_start_df = content_start_df.sort_values(by='Count', ascending=False)
    content_end_df = content_end_df.sort_values(by='Count', ascending=False)
    # Take the first 50 rows and drop the rest:
    content_start_df = content_start_df.head(50).reset_index(drop=True)
    content_end_df = content_end_df.head(50).reset_index(drop=True)
    print(total_rows_df)
    print(total_cols_df)
    print(type_df)
    print(domain_df)
    print(content_start_df)
    print(content_end_df)
    if output_file is not None:
        total_rows_df.to_csv(output_file, mode='w', index=False, header=True)
        total_cols_df.to_csv(output_file, mode='a', index=False, header=False)
        type_df.to_csv(output_file, mode='a', index=False, header=True)
        domain_df.to_csv(output_file, mode='a', index=False, header=True)
        if content_start_df.iloc[0,0] != "":
            content_start_df.to_csv(output_file, mode='a', index=False, header=True)
        else:
            pd.DataFrame([['NONE', 'NONE']], columns=['Content start', 'Count']).to_csv(output_file, mode='a', index=False, header=True)
        if content_end_df.iloc[0,0] != "":
            content_end_df.to_csv(output_file, mode='a', index=False, header=True)
        else:
            pd.DataFrame([['NONE', 'NONE']], columns=['Content end', 'Count']).to_csv(output_file, mode='a', index=False, header=True)
        print("Statistics added to csv file")
    return content_start_df, content_end_df


def clean_content(old_filename: str, new_filename: str, df_start: pd.DataFrame, df_end: pd.DataFrame):
    with h5py.File(old_filename, 'r') as read_store, h5py.File(new_filename, 'w') as write_store:
        # Get the original data and create a new dataset:
        data = read_store['data']
        new_data = write_store.create_dataset('data', shape=data.shape, dtype=data.dtype)
        new_data[0] = data[0]
        # Copy the data to the new dataset, but remove the words that are in the DataFrame:
        for i in range(1, data.shape[0]):
            new_data[i] = data[i]
            decoded = new_data[i, COLS['content']].decode('utf-8')
            for w in df_start['Content start']:
                if decoded.startswith(w):
                    new_data[i, COLS['content']] = decoded[len(w):]
            for w in df_end['Content end']:
                if decoded.endswith(w):
                    new_data[i, COLS['content']] = decoded[:-len(w)]


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
    # Copy the raw file to a new file:
    remove_file(path+"raw_copy.h5")
    shutil.copyfile(path+"raw.h5", path+"raw_copy.h5")
    
    statistics(path+"raw_copy.h5", output_file=path+"statistics.csv")
    
    remove_unwanted_rows(data_filename=path+"raw_copy.h5", retained_filename=path+"retained.h5", removed_filename=path+"removed.h5")
    
    remove_file(path+"retained_tmp.h5")
    shutil.copyfile(path+"retained.h5", path+"retained_tmp.h5")
    
    # Clean the data:
    clean_cnt = 0
    while clean_cnt < CLEAN_COUNT:
        clean_cnt += 1
        # Get the statistics:
        df_start, df_end = statistics(path+"retained_tmp.h5")
        # If the dataframes are empty, break the loop:
        if df_start.iloc[0,0] == "" and df_end.iloc[0,0] == "":
            break
        clean_content(old_filename=path+"retained_tmp.h5", new_filename=path+"retained_tmp_cleaned.h5", df_start=df_start, df_end=df_end)
        # Remove the old file and rename the new file so it can be used again:
        remove_file(path+"retained_tmp.h5")
        os.rename(path+"retained_tmp_cleaned.h5", path+"retained_tmp.h5")
    
    remove_file(path+"retained_cleaned.h5")
    os.rename(path+"retained_tmp.h5", path+"retained_cleaned.h5")
    statistics(path+"retained_cleaned.h5", output_file=path+"statistics_cleaned.csv")
    
    shuffle_h5(old_filename=path+"retained_cleaned.h5", new_filename=path+"retained_shuffled.h5")
    h5_to_csv(h5_filename=path+"retained.h5", csv_filename=path+"retained.csv")
    h5_to_csv(h5_filename=path+"retained_shuffled.h5", csv_filename=path+"retained_shuffled.csv")
    h5_to_csv(h5_filename=path+"removed.h5", csv_filename=path+"removed.csv")

if __name__ == '__main__':
    run(sample=SAMPLE)
