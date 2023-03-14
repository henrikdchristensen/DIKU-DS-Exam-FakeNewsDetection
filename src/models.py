
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import pipeline as pp

from sklearn.feature_extraction.text import TfidfVectorizer



def Vectorize():
    cleaned_data = pp.apply_pipeline("../datasets/sample/news_sample.csv", [
        (pp.binary_labels(), 'type'),
        (pp.Clean_data(), 'content')
    ], get_batch=True)
    cleaned_data.to_csv("../datasets/sample/news_sample_cleaned.csv", index=False)

    vect = TfidfVectorizer()
    content_tfidf = vect.fit_transform(cleaned_data['content'])
    content_tfidf_df = pd.DataFrame(content_tfidf.todense(),columns = vect.get_feature_names_out())
    
    content_tfidf_df.to_csv("../datasets/sample/news_sample_tfidf.csv", index=False)


Vectorize()