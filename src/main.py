import pandas as pd
import preprocessing

file = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"


def main():
    df = pd.read_csv(file, usecols=['content'])
    pd.set_option('display.max_colwidth', None)
    df = preprocessing.clean_text(df)
    print(df.loc[1])


if __name__ == '__main__':
    main()
