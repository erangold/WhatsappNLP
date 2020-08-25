import numpy as np
import random

from os import path
import torch

from config import config
from models.checkpoint import save_checkpoint


def train_model(dataset_iter, decoder, optimizer, criterion, checkpoint_dir, device, max_epochs, start_epoch = 0):
    decoder.to(config['device'])

    losses = []
    # with tqdm(range(start_epoch + 1, max_epochs + 1)) as t_epochs:
    for t_epochs in [range(start_epoch + 1, max_epochs + 1)]:  # If we don't want to use tqdm because it looks like shit in pycharm
        n_batches = len(dataset_iter)
        for n_epoch in t_epochs:
            decoder.train()
            loss = 0
            # with tqdm(dataset_iter) as t_batches:
            for t_batches in [dataset_iter]:  # If we don't want to use tqdm because it looks like shit in pycharm
                for batch in t_batches:
                    full_message, full_message_lengths = batch.message
                    input_message = full_message[:-1]
                    output_message = full_message[1:]
                    input_lengths = full_message_lengths - 1

                    loss += train(input_message, input_lengths, output_message, decoder, optimizer, criterion)
            loss /= n_batches
            losses.append(loss)
            print(f"Epoch {n_epoch} -- average batch loss: {loss}")

            # Save checkpoint
            checkpoint = {
                "n_epoch": n_epoch,
                "model": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": loss
            }
            save_checkpoint(checkpoint, path.join(checkpoint_dir, "checkpoint.pt"))
            if n_epoch % 10 == 0:
                save_checkpoint(checkpoint, path.join(checkpoint_dir, f"checkpoint_{n_epoch}.pt"))
    return losses


def train(input_tensor, input_lengths, output_tensor, decoder, decoder_optimizer, criterion):
    decoder_optimizer.zero_grad()

    input_length = input_lengths[0]

    # Teacher forcing: Feed the target as the next input
    decoder_output, _ = decoder(input_tensor, input_lengths)
    loss = criterion(decoder_output, output_tensor.squeeze()[:input_length]) # Teacher forcing

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / input_length.item()


def evaluate(decoder, corpus_vocab, max_length, device):
    decoder.to(device)

    init_token = corpus_vocab.stoi[config['SOS_TOKEN']]
    end_token = corpus_vocab.stoi[config['EOS_TOKEN']]
    sentence_tokens = [[init_token]]
    corpus_tokens = list(range(len(corpus_vocab)))

    with torch.no_grad():
        while sentence_tokens[-1][0] != end_token and len(sentence_tokens) < max_length:
            sentence_tensor = torch.LongTensor(sentence_tokens).to(device)
            sentence_len_tensor = torch.LongTensor([len(sentence_tokens)]).to(device)
            output_log_probs, _ = decoder(sentence_tensor, sentence_len_tensor)
            output_probs = output_log_probs.exp()
            next_token = random.choices(corpus_tokens, weights = output_probs[-1], k = 1)
            sentence_tokens.append(next_token)

    sentence = [corpus_vocab.itos[t[0]] for t in sentence_tokens]
    return sentence


def reweight_distribution(original_distribution, temperature):
    distribution=np.log(original_distribution)/temperature
    distribution=np.exp(distribution)
    return distribution/np.sum(distribution)