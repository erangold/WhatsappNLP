import torch

def save_checkpoint(checkpoint, checkpoint_path):
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['n_epoch'], checkpoint['train_loss']