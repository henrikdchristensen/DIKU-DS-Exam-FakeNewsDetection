import pandas as pd
import warnings

TYPES = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 'bias', 
         'satire', 'state', 'reliable', 'clickbait', 'political']
ROWS_PR_ITERATION = 20000

def get_dataset(input_filename: str = None, output_filename: str = None, size: int = None, remove_unwanted: bool = True) -> pd.DataFrame:
    print("\nGetting dataset...")
    s = 0
    df = pd.DataFrame()
    for chunk in pd.read_csv(input_filename, encoding='utf-8', chunksize=ROWS_PR_ITERATION, lineterminator='\n'):
        if remove_unwanted:
            # Remove rows which have empty content or start with 'ERROR':
            chunk.drop(chunk[chunk['content'].eq('') | chunk['content'].str.startswith('ERROR')].index, inplace=True)
            # Remove rows which does not have a type in types_to_keep:
            chunk.drop(chunk[~chunk['type'].isin(TYPES)].index, inplace=True)
        s += chunk.shape[0]
        # If the size of the dataframe is larger than the size we want, remove the extra rows
        if s > size:
            chunk = chunk.iloc[:size - s]
            s = size
        # Concatenate the chunk to the dataframe
        df = pd.concat([df, chunk], ignore_index=True)
        # If the size of the dataframe is equal to the size we want, break out of the loop
        if s == size:
            break
    if s < size:
        warnings.warn(f'WARNING: The dataset is smaller than the size specified. Size: {s}')
    if output_filename:
        df.to_csv(output_filename, index=False)
    print("\nDataset created")
    return df

def remove_similar_content_in_start_and_end(df: pd.DataFrame, words_compare: int = 10, min_similar: int = 10, max_iterations: int = -1) -> pd.DataFrame:
    words_before = df['content'].str.split().apply(len).sum()
    iteration = 0
    while iteration < max_iterations or max_iterations == -1:
        iteration += 1
        # Split the content by words and get the most common words in the start and end
        start_words = df['content'].str.split().str[:words_compare].apply(tuple).value_counts()
        # Filter out the common words that don't meet the minimum similarity threshold
        start_words = start_words[start_words >= min_similar]
        # Filter out the common words that are shorter than the word comparison length
        start_words = start_words[start_words.index.map(lambda x: len(x) >= words_compare)]
        # Print the number of common words found
        print(f"Removing similar content in the start. Iteration: {iteration}, num of {words_compare}-pair-words similar: {start_words.size}, similar in total: {start_words.sum()}".ljust(200), end='\r')
        # If there are no more common words to remove, break out of the loop
        if start_words.empty:
            break
        # Remove the common words from the start of each content
        df['content'] = df['content'].apply(lambda c: ' '.join(c.split()[words_compare:]) if any(c.startswith(w) for w in start_words.index if len(w) >= words_compare) else c)
    iteration = 0
    while iteration < max_iterations or max_iterations == -1:
        iteration += 1
        # Split the content by words and get the most common words in the start and end
        end_words = df['content'].str.split().str[-words_compare:].apply(tuple).value_counts()
        # Filter out the common words that don't meet the minimum similarity threshold
        end_words = end_words[end_words >= min_similar]
        # Filter out the common words that are shorter than the word comparison length
        end_words = end_words[end_words.index.map(lambda x: len(x) >= words_compare)]
        # Print the number of common words found
        print(f"Removing similar content in the end. Iteration: {iteration}, num of {words_compare}-pair-words similar: {end_words.size}, similar in total: {end_words.sum()}".ljust(200), end='\r')
        # If there are no more common words to remove, break out of the loop
        if end_words.empty:
            break
        # Remove the common words from the end of each content
        df['content'] = df['content'].apply(lambda c: ' '.join(c.split()[:-words_compare]) if any(c.endswith(w) for w in end_words.index if len(w) >= words_compare) else c)
    words_after = df['content'].str.split().apply(len).sum()
    print(f'\nNumber of words before: {words_before} and after: {words_after}. Difference: {words_before - words_after}')
    return df


def run():        
    choice = input("Press 's' for sample or 'l' for large dataset or 'x' to Exit: ")
    if choice == 'x':
        return
    elif choice == 's':
        path = "../datasets/sample/"
    elif choice == 'l':
        path = "../datasets/large/"
    else:
        print("Invalid choice - exiting")
        return
    input_filename = path+"shuffled.csv"
    choice = input("Output to file? Press 'y' for yes or 'n' for no or 'x' to Exit: ")
    if choice == 'x':
        return
    elif choice == 'y':
        output_filename = path+"dataset.csv"
    elif choice == 'n':
        output_filename = None
    else:
        print("Invalid choice - exiting")
        return
    choice = input("Size to generate. Press 'x' to Exit: ")
    if choice == 'x':
        return
    size = int(choice)
    choice = input("Remove unwanted? Press 'y' for yes or 'n' for no or 'x' to Exit: ")
    if choice == 'x':
        return
    elif choice == 'y':
        remove_unwanted = True
    elif choice == 'n':
        remove_unwanted = False
    else:
        print("Invalid choice - exiting")
        return
    df = get_dataset(input_filename=input_filename, output_filename=output_filename, size=size, remove_unwanted=remove_unwanted)
    #remove_similar_content_in_start_and_end(df)
    #df = remove_unwanted_rows(df, TYPES)

if __name__ == '__main__':
    run()