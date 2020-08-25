import os
from os import path
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim, nn

from torchsummary import summary
from config import config
from dataset.normalize import normalize_dataset, normalize_contractions
from models.checkpoint import load_checkpoint
from models.rnn.decoder import DecoderRNN
from models.rnn.train import evaluate, train_model
from dataset.loader import load_mentions_dataset, get_mentions_dataset_iterator


def main():
    raw_file_path = path.join(config['base_path'], "data", config['raw_dataset'])
    csv_file_path = path.join(config['base_path'], "data", config['normalized_dataset'])
    checkpoint_dir = path.join(config['base_path'], "checkpoint")

    if config['mode'] == 'normalize':
        print("Normalizing dataset")
        normalize_dataset(raw_file_path, csv_file_path, encoding=config['encoding'])
        return

    embedding_file_path = None
    if config['embedding_mode'] != 'random':
        embedding_file_path = path.join(config['base_path'], "embeddings", config['precomputed_embedding_file'])

    if not os.path.exists(csv_file_path):
        raise Exception("CSV file not found!")
    elif embedding_file_path and not os.path.exists(embedding_file_path):
        raise Exception("Precomputed embedding file not found!")

    print("Loading dataset")
    embedding_vectors, corpus_size, dataset = load_mentions_dataset(csv_file_path, {}, embedding_file_path, fix_len = config['max_sentence_len'])
    dataset_iter = get_mentions_dataset_iterator(dataset, batch_size=config['batch_size'])
    print("Done...")

    if config['embedding_mode'] == 'precomputed':
        print(f"Using precomputed embedding (path: {embedding_file_path})")
        kwargs = {"num_embeddings": corpus_size, "word_embedding" : embedding_vectors}
    elif config['embedding_mode'] == 'precomputed-light':
        # TODO: Filter only embedding of words in the corpus
        kwargs = {"num_embeddings": corpus_size, "word_embedding" : embedding_vectors}
    else:
        kwargs = {"num_embeddings" : corpus_size, "embedding_dim": config['embedding_dim'], "hidden_size": config['hidden_size']}

    print("\nInitializing model")
    decoder = DecoderRNN(config['embedding_mode'], **kwargs)
    optimizer = optim.SGD(decoder.parameters(), lr=config['learning_rate'])

    start_epoch = 0
    if not config['cold_start']:
        checkpoint_file = path.join(checkpoint_dir, "checkpoint.pt")
        decoder, optimizer, start_epoch, loss = load_checkpoint(checkpoint_file, decoder, optimizer)
        print(f"Loading precomputed model, starting at epoch {start_epoch}")
    decoder = decoder.to(config['device'])
    print(decoder)

    if config['mode'] == 'eval':
        print("\nEvaluating model")
        res = ""
        while res != 'e':
            ds_vocab = dataset.fields['message'].vocab
            max_length = 15
            sentence_tokens = evaluate(decoder, ds_vocab, max_length, config['device'])
            print(' '.join(sentence_tokens))
            print("\nInsert 'e' to terminate")
            res = input()
        return
    elif config['mode'] == 'train':
        print("\nTraining model")
        criterion = nn.NLLLoss()

        losses = train_model(
            dataset_iter = dataset_iter,
            decoder = decoder,
            optimizer = optimizer,
            criterion = criterion,
            checkpoint_dir = checkpoint_dir,
            start_epoch = start_epoch,
            device = config['device'],
            max_epochs = config['max_epochs']
        )

        fig, ax = plt.subplots()
        ax.plot(range(config['max_epochs']), losses)
        ax.set_xlabel("n_epoch")
        ax.set_ylabel("loss")
        ax.set_xticks(range(config['max_epochs']))
        plt.show()
        return


if __name__ == '__main__':
    main()