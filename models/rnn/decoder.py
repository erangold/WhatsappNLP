import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq

class DecoderRNN(nn.Module):
    def __init__(self, embedding_mode, word_embedding = None, num_embeddings = None, embedding_dim = None, hidden_size = None):
        super(DecoderRNN, self).__init__()

        if embedding_mode == 'precomputed' or embedding_mode == 'precomputed-light':
            output_size = num_embeddings
            self.hidden_size = embedding_dim = word_embedding.size()[-1]
            self.embedding = nn.Embedding.from_pretrained(word_embedding)
        elif embedding_mode == 'random':
            output_size = num_embeddings
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
            self.hidden_size = hidden_size

        self.lstm = nn.LSTM(embedding_dim, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, input_lens, rnn_init = None):
        batch_size = input.shape[1]
        embeds = self.embedding(input)
        embeds = F.relu(embeds)

        try:
            # Pack
            packed_x = packseq(embeds, list(input_lens), batch_first=False)
        except ValueError as e:
            print(input_lens)
            raise e
            sys.exit(0)

        if not rnn_init:
            rnn_init = self.initialize_rnn_hidden(1, batch_size, input.device)

        packed_output, hidden = self.lstm(packed_x, rnn_init)
        output, input_sizes = padseq(packed_output, batch_first=False)
        output_softmax = self.softmax(self.out(output).squeeze(dim = 1))
        return output_softmax, hidden

    def initialize_rnn_hidden(self, nlayers, batch_size, device):
        return (self.init_hidden(nlayers, batch_size, device),
                    self.init_hidden(nlayers, batch_size, device))

    def init_hidden(self, nlayers, batch_size, device):
        return torch.zeros((nlayers, batch_size, self.hidden_size), device = device)