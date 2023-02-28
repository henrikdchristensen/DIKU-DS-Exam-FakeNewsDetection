from multiprocessing import reduction
import pandas as pd
import Preprocessing as ps
import os

#file = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"
file = "datasets/news_sample.csv"

# Set current directory one level up:
os.chdir("..")

def main():
    pipeline = pd.read_csv(file, usecols=['id'])
    content = pd.read_csv(file, usecols=['content'])
    pd.set_option('display.max_colwidth', None)
    cleaned = preprocessing.clean_text(content)
    tokenized = preprocessing.tokenize_text(cleaned)
    without_stopwords = tokenized.applymap(lambda x: preprocessing.remove_stopwords(x))
    stemmed = without_stopwords.applymap(lambda x: preprocessing.stem(x))
    pipeline = pipeline.assign(content = content, cleaned = cleaned, tokenized = tokenized, without_stopwords = without_stopwords, stemmed = stemmed)

#Compute the reduction rate of the vocabulary size after removing stopwords.
#Compute the reduction rate of the vocabulary size after stemming.

    preprocessor = preprocessing.Preprocessing()
    word_count_with_stopwords = preprocessor.get_word_count(tokenized)
    word_count_without_stopwords = preprocessor.get_word_count(without_stopwords)
    print("wc before removing stopwords: " + word_count_with_stopwords)
    print("wc after removing stopwords: " + word_count_without_stopwords)

    word_count_after_stemming = preprocessor.get_word_count(stemmed)
    print("wc after stemming: " + word_count_after_stemming)

    reduction_rate_stopwords = 1 - word_count_without_stopwords/word_count_with_stopwords
    reduction_rate_stemming = 1 - word_count_after_stemming/word_count_without_stopwords
    print("red. rate stopwords: " + reduction_rate_stopwords)
    print("red. rate stemming: " +reduction_rate_stemming)



if __name__ == '__main__':
    main()
