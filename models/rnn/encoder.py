import torch
from torch import nn

class EncoderRNN(nn.Module):
    def __init__(self, word_embedding_vecs):
        super(EncoderRNN, self).__init__()
        self.hidden_size = word_embedding_vecs.dim

        self.embedding = nn.Embedding.from_pretrained(word_embedding_vecs)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)