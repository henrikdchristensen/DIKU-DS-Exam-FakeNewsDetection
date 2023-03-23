import pandas as pd

TYPES = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 'bias', 
         'satire', 'state', 'reliable', 'clickbait', 'political']


def get_pandas_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    return df

def remove_similar_content_in_start_and_end(df: pd.DataFrame, words_compare: int = 5, min_similar: int = 5, max_iterations: int = -1) -> pd.DataFrame:
    print("\nRemoving similar content in the start and end...")
    iterations = max_iterations
    while iterations > 0 or iterations == -1:
        if iterations != -1:
            iterations -= 1
        start_words = df['content'].str.split().str[:words_compare].str.join(' ').value_counts().sort_values(ascending=False)
        start_words = start_words[start_words > min_similar].index.tolist()
        end_words = df['content'].str.split().str[-words_compare:].str.join(' ').value_counts().sort_values(ascending=False)
        end_words = end_words[end_words > min_similar].index.tolist()
        if start_words[0] == '' and end_words[0] == '':
            break
        if start_words[0] != '':
            print(f"\nStart words being removed: {start_words}\n")
            for w in start_words:
                df['content'] = df['content'].apply(lambda x: ' '.join(x.split()[words_compare:]) if x.startswith(w) else x)
        if end_words[0] != '':
            print(f"End words being removed: {end_words}\n")
            for w in end_words:
                df['content'] = df['content'].apply(lambda x: ' '.join(x.split()[:-words_compare]) if x.endswith(w) else x)
    return df


def remove_unwanted_rows(df: pd.DataFrame, types_to_keep: list) -> pd.DataFrame:
    print("\nRemoving unwanted content...")
    print(f"Original shape: {df.shape}")
    # Remove rows which have empty content or start with 'Error':
    df.drop(df[df['content'].eq('') | df['content'].str.startswith('Error')].index, inplace=True)
    # Remove rows which does not have a type in types_to_keep:
    df.drop(df[~df['type'].isin(types_to_keep)].index, inplace=True)
    print(f"Cleaned shape: {df.shape}")
    return df
    
df = get_pandas_df("../datasets/sample/raw.csv")
#remove_similar_content_in_start_and_end(df)
remove_unwanted_rows(df, TYPES)