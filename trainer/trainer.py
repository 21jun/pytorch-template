import numpy as np
import torch
from tqdm import tqdm


class Trainer:

    def __init__(self, model, data_loader, valid_data_loader,
                 criterion, optimizer, epochs, device,
                 metric_ftns=None, lr_scheduler=None):

        self.device = device
        self.model = model.to(device)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler

    def train(self, do_valid=True):

        best_acc = 0.0

        for epoch in range(1, self.epochs + 1):

            print("epoch", "|", epoch)
            info = {}
            train_loss, train_acc = self._train_epoch(epoch)
            info['train_loss'] = train_loss
            info['train_acc'] = train_acc

            if do_valid:
                valid_loss, valid_acc = self._valid_epoch(epoch)
                info['valid_loss'] = valid_loss
                info['valid_acc'] = valid_acc
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    # save this file

            self._progress(info, do_valid)

    def _train_epoch(self, epoch):

        self.model.train()

        train_loss = 0.0
        train_total = 0
        train_correct = 0

        for batch_idx, (data, target) in tqdm(enumerate(self.data_loader)):
            self.optimizer.zero_grad()

            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            _, predictions = output.max(1)
            correct = predictions.eq(target).sum().item()

            train_loss += batch_loss
            train_total += target.size(0)
            train_correct += correct

        train_acc = (train_correct / train_total) * 100

        return train_loss, train_acc

    def _valid_epoch(self, epoch):

        self.model.eval()

        valid_loss = 0.0
        valid_total = 0
        valid_correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.valid_data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                batch_loss = loss.item()
                _, predictions = output.max(1)
                correct = predictions.eq(target).sum().item()

                valid_loss += batch_loss
                valid_total += target.size(0)
                valid_correct += correct

        valid_acc = (valid_correct / valid_total) * 100

        return valid_loss, valid_acc

    def _progress(self, info, do_valid):
        print("train_loss :", info['train_loss'],
              "train_acc :", info['train_acc'])
        if do_valid:
            print("valid_loss :", info['valid_loss'],
                  "valid_acc :", info['valid_acc'])
