from dataset.tokenizer import tokenize_only_words
from dataset.whatsapp_formatter import read_raw_transcript
from rftokenizer import RFTokenizer
import pandas as pd

rf_tokenizer = RFTokenizer('heb')


def normalize_dataset(input_file_path, output_file_path, encoding, split_punc = True, split_contractions = True):
    # dataset_df = read_raw_transcript(input_file_path, encoding=encoding)
    dataset_df = pd.read_csv(input_file_path[:-3] + 'csv')

    if split_punc:
        print("Extracting punctuation symbols")
        dataset_df['message'] = dataset_df['message'].apply(normalize_punc)

    if split_contractions:
        print("Splitting lingual contractions")
        dataset_df['message'] = dataset_df['message'].apply(normalize_contractions)

    dataset_df.to_csv(output_file_path, encoding=encoding, index=False)


def normalize_punc(s):
    tokens = tokenize_only_words(s)
    return ' '.join(tokens)


def normalize_contractions(s):
    tokens = rf_tokenizer.rf_tokenize(s.split(), sep=' ')
    return ' '.join(tokens)