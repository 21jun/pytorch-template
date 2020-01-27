import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader.data_loaders import MnistDataLoader
from model.model import MnistModel
from trainer.trainer import Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.set_device(0)
    print("cuda:", torch.cuda.current_device())


data_loader = MnistDataLoader(
    data_dir='data/', batch_size=1024, shuffle=True, validation_split=0.3, num_workers=0)

valid_data_loader = data_loader.split_validation()

model = MnistModel()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

trainer = Trainer(model=model, data_loader=data_loader, valid_data_loader=valid_data_loader, criterion=criterion,
                  optimizer=optimizer, epochs=5, device=device)

trainer.train(do_valid=True)
