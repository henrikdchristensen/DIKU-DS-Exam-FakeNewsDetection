import pandas as pd
import preprocessing
import os

#file = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"
file = "datasets/news_sample.csv"

# Set current directory one level up:
os.chdir("..")


def main():
    df = pd.read_csv(file, usecols=['content'])
    pd.set_option('display.max_colwidth', None)
    df = preprocessing.clean_text(df)
    df = preprocessing.tokenize_text(df)
    print(df.loc[1])


if __name__ == '__main__':
    main()
