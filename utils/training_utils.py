import os
import torch
import numpy as np


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, checkpoint_path)
    return checkpoint_path

def adjust_accuracy(x, chance):
    possible = 1 - chance
    actual   = x - chance
    adjusted = actual / possible

    return adjusted