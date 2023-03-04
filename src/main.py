from multiprocessing import reduction
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


    pipeline = pd.read_csv(file, usecols=['id'])
    content = pd.read_csv(file, usecols=['content'])
    pd.set_option('display.max_colwidth', None)
    cleaned = ps.clean_text(content)
    tokenized = ps.tokenize_text(cleaned)
    without_stopwords_nltk = tokenized.applymap(lambda x: ps.remove_stopwords_nltk(x))

    stop_words_tail = ps.Exploration.get_stopwords(tokenized, 0, 2e-3)
    stop_words_notail = ps.Exploration.get_stopwords(tokenized, 3e-5, 2e-3)
    without_stopwords_freq_tail = tokenized.applymap(lambda x: ps.remove_stopwords_freq(x, stop_words_tail))
    without_stopwords_freq_notail = tokenized.applymap(lambda x: ps.remove_stopwords_freq(x, stop_words_notail))
    
    stemmed = without_stopwords_nltk.applymap(lambda x: ps.stem(x))
    pipeline = pipeline.assign(content = content, cleaned = cleaned, tokenized = tokenized, without_stopwords = without_stopwords_nltk, stemmed = stemmed)

#Compute the reduction rate of the vocabulary size after removing stopwords.
#Compute the reduction rate of the vocabulary size after stemming.

    
    word_count_with_stopwords = ps.Exploration.get_word_info(tokenized)
    word_count_without_stopwords = ps.Exploration.get_word_info(without_stopwords_nltk)
    word_count_without_stopwords_freq_tail = ps.Exploration.get_word_info(without_stopwords_freq_tail)
    word_count_without_stopwords_freq_notail = ps.Exploration.get_word_info(without_stopwords_freq_notail)
    print(f"vocab before removing stopwords: {len(word_count_with_stopwords)}")
    print(f"vocab after nltk stopwords: {len(word_count_without_stopwords)}")
    print(f"vocab after freq stopwords tail: {len(word_count_without_stopwords_freq_tail)}")
    print(f"vocab after freq stopwords no tail: {len(word_count_without_stopwords_freq_notail)}")

    word_count_after_stemming = ps.Exploration.get_word_info(stemmed)
    print(f"wc after stemming: {len(word_count_after_stemming)}")

    reduction_rate_stopwords = 1 - len(word_count_without_stopwords)/len(word_count_with_stopwords)
    #reduction_rate_stopwords_freq_tail = 1 - len(word_count_without_stopwords)/len(word_count_with_stopwords)
    #reduction_rate_stopwords_freq_notail = 1 - len(word_count_without_stopwords)/len(word_count_with_stopwords)
    reduction_rate_stemming = 1 - len(word_count_after_stemming)/len(word_count_without_stopwords)
    print(f"red. rate stopwords: {reduction_rate_stopwords}")
    print(f"red. rate stemming: {reduction_rate_stemming}")
    
    url_count = ps.Exploration.countItems(cleaned)
    #source_dist = ps.Exploration.sourceDistribution(cleaned)

    #print(source_dist)
    # date_count = ps.Exploration.countDates(df)
    # number_count = ps.Exploration.countNumbers(df)

    # print("URLs: ", url_count, "Dates: ", date_count, "Numbers: ", number_count)

    print("URLs: ", url_count)


if __name__ == '__main__':
    main()
