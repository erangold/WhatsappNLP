import torch
from torchtext import vocab, data
from os import path

from config import config
from dataset.tokenizer import tokenize_only_words as heb_tokenize


def load_mentions_dataset(chat_dataset_path, csv_reader_params, word_embedding_file, fix_len = None):
    word_embedding_vectors = None
    if word_embedding_file:
        if word_embedding_file.endswith(".pt"):
            word_embedding_file = word_embedding_file[:-3]

        word_embedding_vectors = vocab.Vectors(
            word_embedding_file,
            cache = path.dirname(word_embedding_file))

    message_field = data.ReversibleField(sequential=True,
                                         # tokenize=heb_tokenize,
                                         fix_length = fix_len,
                                         init_token=config['SOS_TOKEN'],
                                         eos_token=config['EOS_TOKEN'],
                                         pad_first=False,
                                         include_lengths=True
                                         )

    fields = {
        'message': ('message', message_field)
    }

    dataset = data.TabularDataset(
        path=chat_dataset_path,
        format='csv',
        csv_reader_params=csv_reader_params,
        skip_header=False,
        fields=fields
    )

    if word_embedding_file:
        message_field.build_vocab(dataset, vectors=word_embedding_vectors)
    else:
        message_field.build_vocab(dataset)
    corpus_size = len(message_field.vocab)
    return message_field.vocab.vectors, corpus_size, dataset


def get_mentions_dataset_iterator(mentions_dataset, batch_size):
    dataset_iter = data.BucketIterator(mentions_dataset,
                                       batch_size=batch_size,
                                       sort_key=lambda x: (len(x.message)),
                                       sort_within_batch=True,
                                       shuffle=False,
                                       device=config['device'])
    return dataset_iter