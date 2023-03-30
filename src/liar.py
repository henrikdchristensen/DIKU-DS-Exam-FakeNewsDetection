import pipeline as pp
import filehandling as fh


headers = ['id', 'label', 'statement', 'subjects', 'speaker', 'speaker_job', 'state_info',
           'party', 'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context']


def create_dataset(file, new_file):
    pp.apply_pipeline(file, [
        # Binary labels
        (pp.Binary_labels_LIAR(), 'label', 'label_binary'),
        # Clean content
        (pp.Clean_id_LIAR(), 'id', 'id'),
        (pp.Clean_data(), 'statement', 'statement_cleaned'),
        (pp.Tokenizer(), 'statement_cleaned'),
        (pp.Remove_stopwords(), 'statement_cleaned'),
        (pp.Stem(), "statement_cleaned"),
        (pp.Combine_Content(), 'statement_cleaned', 'statement_combined'),
        (pp.Sentence_analysis(), 'statement_combined', 'sentence_analysis'),
    ],
        new_file=new_file,
        progress_bar=True,
    )


if __name__ == '__main__':
    fh.tsv_to_csv('../datasets/liar_dataset/raw/train.tsv',
                  '../datasets//liar_dataset/cleaned/train.csv', headers=headers)
    fh.tsv_to_csv('../datasets/liar_dataset/raw/valid.tsv',
                  '../datasets//liar_dataset/cleaned/valid.csv', headers=headers)
    fh.tsv_to_csv('../datasets/liar_dataset/raw/test.tsv',
                  '../datasets//liar_dataset/cleaned/test.csv', headers=headers)
    fh.combine_csv_files(['../datasets/liar_dataset/cleaned/train.csv', '../datasets/liar_dataset/cleaned/valid.csv',
                          '../datasets/liar_dataset/cleaned/test.csv'], '../datasets/liar_dataset/cleaned/combined.csv')
    create_dataset('../datasets/liar_dataset/cleaned/combined.csv',
                   '../datasets/liar_dataset/cleaned/combined_cleaned.csv')
