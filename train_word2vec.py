from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader.data_loaders import Word2vecDataLoader
from model.model import Word2VecModel
from trainer import Word2vecTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.set_device(0)
    print("cuda:", torch.cuda.current_device())


data_loader = Word2vecDataLoader(
    data_dir='data/Word2vec/input.txt', batch_size=32, shuffle=True, validation_split=0.3, num_workers=0)

valid_data_loader = data_loader.split_validation()

skip_gram_model = Word2VecModel(data_loader.vocab_size, 2)
initial_lr = 0.001
optimizer = optim.SparseAdam(skip_gram_model.parameters(), lr=initial_lr)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, len(data_loader))
save_dir = Path('saved/Word2vec')
resume_path = None

trainer = Word2vecTrainer(model=skip_gram_model, data_loader=data_loader,
                          valid_data_loader=valid_data_loader, criterion=None,
                          optimizer=optimizer, lr_scheduler=lr_scheduler,
                          epochs=5, device=device, save_dir=save_dir)

trainer.train(do_valid=False, do_save=False)
