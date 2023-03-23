import pandas as pd

def get_pandas_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    return df

def remove_start_end_content(df: pd.DataFrame, words_compare: int = 5, min_similar: int = 5, max_iterations: int = -1) -> pd.DataFrame:
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
    
df = get_pandas_df("../datasets/sample/raw.csv")
remove_start_end_content(df)