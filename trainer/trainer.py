import numpy as np
import torch
from tqdm import tqdm
from base import BaseTrainer


class MnistTrainer(BaseTrainer):

    def __init__(self, model, data_loader, valid_data_loader,
                 criterion, optimizer, epochs, device, save_dir,
                 metric_ftns=None, lr_scheduler=None, resume_path=None):

        super().__init__(model, data_loader, valid_data_loader,
                         criterion, optimizer, epochs, device, save_dir, metric_ftns,
                         lr_scheduler, resume_path)

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


class Word2vecTrainer(BaseTrainer):

    def __init__(self, model, data_loader, valid_data_loader,
                 criterion, optimizer, epochs, device, save_dir,
                 metric_ftns=None, lr_scheduler=None, resume_path=None):

        super().__init__(model, data_loader, valid_data_loader,
                         criterion, optimizer, epochs, device, save_dir, metric_ftns,
                         lr_scheduler, resume_path)

    def train(self, do_valid=True, do_save=False):

        best_acc = 0.0

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs + 1):

            print("epoch", "|", epoch)
            info = {}
            train_loss = self._train_epoch(epoch)
            info['train_loss'] = train_loss

            if do_valid:
                valid_loss = self._valid_epoch(epoch)
                info['valid_loss'] = valid_loss

                if valid_acc > best_acc:
                    print("Best_acc updated : ", best_acc, "->", valid_acc)
                    best_acc = valid_acc

                    if do_save:
                        self._save_checkpoint(epoch)

            self._progress(info)

    def _train_epoch(self, epoch):

        self.model.train()

        train_loss = 0.0

        for batch_idx, sample_batched in tqdm(enumerate(self.data_loader)):
            if len(sample_batched[0]) > 1:
                pos_u = sample_batched[0].to(self.device)
                pos_v = sample_batched[1].to(self.device)
                neg_v = sample_batched[2].to(self.device)

                self.optimizer.zero_grad()
                loss = self.model(pos_u, pos_v, neg_v)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                train_loss = train_loss * 0.9 + loss.item() * 0.1

        return train_loss

    def _valid_epoch(self, epoch):

        self.model.eval()

        valid_loss = 0.0

        with torch.no_grad():
            for batch_idx, (sample_batched) in tqdm(enumerate(self.valid_data_loader)):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                loss = self.model(pos_u, pos_v, neg_v)

                valid_loss = valid_loss * 0.9 + loss.item() * 0.1

        return valid_loss
