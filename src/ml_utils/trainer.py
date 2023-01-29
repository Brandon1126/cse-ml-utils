"""Trainer Module to assist with Training.
"""
import os
import torch
import numpy as np
from time import time
from typing import Callable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.models import resnet18


class Trainer:
    """Trainer Class
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        loss_func: Callable,  # from torch.nn.functional.*
        optimizer: torch.optim.Optimizer,
        max_run_time: float,
        snapshot_name: str,
    ) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        print(f"Model loaded on device: {next(model.parameters()).device}")
        self.train_data = train_data
        self.valid_data = valid_data
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 0.98, last_epoch=-1, verbose=False)
        # Hours to seconds, training will stop at this time
        self.max_run_time = max_run_time * 60**2
        self.save_path = "training_saves/" + snapshot_name
        self.epochs_run = 0  # current epoch tracker
        self.run_time = 0.0  # current run_time tracker
        self.train_loss_history = list()
        self.valid_loss_history = list()
        self.epoch_times = list()
        self.lowest_loss = np.Inf
        self.train_loss = np.Inf
        self.valid_loss = np.Inf
        # Loading in existing training session if the save destination already exists
        if os.path.exists(self.save_path):
            print("Loading snapshot")
            self._load_snapshot(self.save_path)
        if self.train_loss_history:
            self.train_loss = self.train_loss_history[-1]
            self.valid_loss = self.valid_loss_history[-1]

    def _load_snapshot(self, snapshot_path):
        loc = "cuda:0"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.run_time = snapshot['RUN_TIME']
        self.train_loss_history = snapshot['TRAIN_HISTORY']
        self.valid_loss_history = snapshot['VALID_HISTORY']
        self.epoch_times = snapshot['EPOCH_TIMES']
        self.lowest_loss = snapshot['LOWEST_LOSS']
        print(f"Resuming training from save at Epoch {self.epochs_run}")

    def _calc_validation_loss(self, source, targets) -> float:
        self.model.eval()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        self.model.train()
        return float(loss.item())

    def _run_batch(self, source, targets) -> float:
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _run_epoch(self):
        b_sz = len(next(iter(self.train_data))[0])
        if self.epochs_run % 10 == 0:
            print(
                f"\nEpoch: {self.epochs_run} | Batch_SZ: {b_sz} ", end="")
            print(
                f"| Steps: {len(self.train_data)} ", end="")
            print(
                f"| T_loss: {self.train_loss:.3f} | V_loss: {self.valid_loss:.3f}")
        # self.train_data.sampler.set_epoch(self.epochs_run)
        train_loss = 0
        valid_loss = 0

        # Train Loop
        for source, targets in self.train_data:
            source = source.to(self.device)
            targets = targets.to(self.device)
            train_loss += self._run_batch(source, targets)

        # Calculating Validation loss
        for source, targets in self.valid_data:
            source = source.to(self.device)
            targets = targets.to(self.device)
            valid_loss += self._calc_validation_loss(source, targets)

        # Update loss history & scheduler.step
        self.scheduler.step()
        self.train_loss_history.append(train_loss/len(self.train_data))
        self.valid_loss_history.append(valid_loss/len(self.valid_data))
        self.train_loss, self.valid_loss = self.train_loss_history[-1], self.valid_loss_history[-1]

    def _save_snapshot(self):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "EPOCHS_RUN": self.epochs_run,
            "RUN_TIME": self.run_time,
            "TRAIN_HISTORY": self.train_loss_history,
            "VALID_HISTORY": self.valid_loss_history,
            "EPOCH_TIMES": self.epoch_times,
            "LOWEST_LOSS": self.lowest_loss
        }
        torch.save(snapshot, self.save_path)
        print(
            f"Training snapshot saved after Epoch: {self.epochs_run} | save_name: {self.save_path}")

    def train(self):
        for _ in range(self.epochs_run, self.epochs_run + 1000):
            start = time()
            self._run_epoch()
            elapsed_time = time() - start
            self.run_time += elapsed_time
            self.epoch_times.append(elapsed_time)
            start = time()
            self.epochs_run += 1
            if self.valid_loss_history[-1] < self.lowest_loss:
                self.lowest_loss = self.valid_loss_history[-1]
                self._save_snapshot()
            elapsed_time = time() - start
            self.run_time += elapsed_time
            self.epoch_times[-1] += elapsed_time
            if self.epochs_run % 10 == 1:
                print(
                    f'Current Train Time: {self.run_time//60**2} hours & {((self.run_time%60.0**2)/60.0):.2f} minutes')
            if (self.run_time > self.max_run_time):
                print(
                    f"Training completed -> Total train time: {self.run_time:.2f} seconds")
                break

        # Saving import metrics to analyze training on local machine
        train_metrics = {
            "EPOCHS_RUN": self.epochs_run,
            "RUN_TIME": self.run_time,
            "TRAIN_HISTORY": self.train_loss_history,
            "VALID_HISTORY": self.valid_loss_history,
            "EPOCH_TIMES": self.epoch_times,
            "LOWEST_LOSS": self.lowest_loss
        }
        torch.save(train_metrics, self.save_path[:-3] + "_metrics.pt")


def print_classification_model_test(model: torch.nn.Module,
                                    test_loader: DataLoader,
                                    labels_map: dict,
                                    loss_func: Callable = F.cross_entropy
                                    ) -> None:
    """Prints out model testing stats, only works on classificiation models

    Args:
        model (torch.nn.Module): model to be tested
        test_loader (DataLoader): test data
        labels_map (dict): dictionary map from class index to class name
        loss_func (Callable, optional): loss func. Defaults to F.cross_entropy.
    """
    # initialize lists to monitor test loss and accuracy
    classes_num = 10
    test_loss = 0.0
    class_correct = list(0. for i in range(classes_num))
    class_total = list(0. for i in range(classes_num))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_func(output, target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        for i in range(len(target)):
            label = target[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    t_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(t_loss))

    for i in range(classes_num):
        if class_total[i] > 0:
            percent_correct = 100 * class_correct[i] / class_total[i]
            print(
                f'Test Accuracy of Class: {labels_map[i]:19s}: {percent_correct:3.2f}%',
                f'({np.sum(int(class_correct[i]))}/{np.sum(int(class_total[i]))})')
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' %
                  (labels_map[i]))

    percent_correct = 100.0 * np.sum(class_correct) / np.sum(class_total)
    print(
        f'\nTest Accuracy (Overall): {percent_correct:3.2f}%',
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})')
