import panda as pd

file = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"


def main():
    df = pd.read_csv(file, usecols=['content'])
    pd.set_option('display.max_colwidth', None)
    print(df.loc[0])


if __name__ == '__main__':
    main()
