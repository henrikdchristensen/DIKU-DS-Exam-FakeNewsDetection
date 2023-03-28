import matplotlib.pyplot as plt
import pipeline as pp
import pandas as pd
from textblob import TextBlob
import filehandling as fh
from nltk.corpus import stopwords


headers = ['id', 'label', 'statement', 'subjects', 'speaker', 'speaker_job', 'state_info', 
           'party', 'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context']


def tsv_to_csv_and_set_headers(file: str, new_file: str, headers: list = None):
    df = pd.read_csv(file, delimiter='\t', header=None)
    if headers is not None:
        df.columns = headers
    df.to_csv(new_file, index=False)
    
def combine_csv_files(files: list, new_file: str):
    df = pd.concat([pd.read_csv(f) for f in files])
    df.to_csv(new_file, index=False)

    
class Clean_id(pp.FunctionApplier):
    def function_to_apply(self, id):
        return id.split('.')[0]
    
def clean_dataset(file, new_file):
    stopwords_lst = stopwords.words('english')
    pp.apply_pipeline(file, [
        # Binary labels
        #(pp.Binary_labels(), 'type', 'type_binary'),
        # Clean content
        (Clean_id(), 'id', 'id'),
        (pp.Clean_data(), 'statement', 'statement_cleaned'),
        (pp.Tokenizer(), 'statement_cleaned'),
        (pp.Remove_stopwords(stopwords_lst), 'statement_cleaned'),
        (pp.Stem(), "statement_cleaned"),
        (pp.Combine_Content(), 'statement_cleaned', 'statement_combined'),
        (pp.Sentence_analysis(), 'statement_combined', 'sentence_analysis'),
    ],
        new_file=new_file,
        progress_bar=True,
    )


if __name__ == '__main__':
    tsv_to_csv_and_set_headers('../datasets/liar_dataset/raw/train.tsv', '../datasets//liar_dataset/cleaned/train.csv', headers=headers)
    tsv_to_csv_and_set_headers('../datasets/liar_dataset/raw/valid.tsv', '../datasets//liar_dataset/cleaned/valid.csv', headers=headers)
    tsv_to_csv_and_set_headers('../datasets/liar_dataset/raw/test.tsv', '../datasets//liar_dataset/cleaned/test.csv', headers=headers)
    combine_csv_files(['../datasets/liar_dataset/cleaned/train.csv', '../datasets/liar_dataset/cleaned/valid.csv', '../datasets/liar_dataset/cleaned/test.csv'], '../datasets/liar_dataset/cleaned/combined.csv')
    clean_dataset('../datasets/liar_dataset/cleaned/combined.csv', '../datasets/liar_dataset/cleaned/combined_cleaned.csv')