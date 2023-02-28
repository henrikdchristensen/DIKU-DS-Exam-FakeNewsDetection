import pandas as pd
import Preprocessing as ps
import os

#file = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"
file = "datasets/news_sample.csv"

# Set current directory one level up:
os.chdir("..")


def main():
    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv(file)
    # df = ps.clean_text(df)
    # df = ps.tokenize_text(df)
    # print(ps.Exploration.get_stopwords(df, freq_low = 5e-5, freq_high = 1e-3, print_stopwords_info = True))


    url_count = ps.Exploration.countItems(df)
    source_dist = ps.Exploration.sourceDistribution(df)

    print(source_dist)
    # date_count = ps.Exploration.countDates(df)
    # number_count = ps.Exploration.countNumbers(df)

    # print("URLs: ", url_count, "Dates: ", date_count, "Numbers: ", number_count)

    print("URLs: ", url_count)


# def exploreData() 


if __name__ == '__main__':
    main()
