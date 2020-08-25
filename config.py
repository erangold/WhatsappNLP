import torch

config = {
    'base_path' : r"C:\Users\erangold\Documents\INoteBook\HamutziPEDIA",
    'raw_dataset' : "chat_transcript.txt",
    'normalized_dataset' : "speaker_dataset__גיא_דינר.csv",
    'encoding' : 'utf8',
    'max_sentence_len' : 10,
    'max_epochs' : 60,
    'learning_rate' : 0.1,
    'batch_size' : 1,
    'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'embedding_dim' : 300,
    'hidden_size' : 100,
    'SOS_TOKEN' : "<sos>",
    'EOS_TOKEN' : "<eos>",
    'embedding_mode' : 'precomputed', # {precomputed, precomputed-light, random}
    'precomputed_embedding_file' : "cc.he.300.vec.gz",
    'mode' : "eval", # {train, eval, normalize}
    'cold_start' : False
}
