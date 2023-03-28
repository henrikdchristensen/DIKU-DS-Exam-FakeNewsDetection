import pandas as pd

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