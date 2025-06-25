#!/usr/bin/env python

import torch
import os
import numpy as np


class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=True, delta=0, save_path='.'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model, train_losses, val_losses, optimizer, scheduler, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, train_losses, val_losses, optimizer, scheduler, epoch)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, train_losses, val_losses, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, train_losses, val_losses, optimizer, scheduler, epoch):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_losses': train_losses,
            'test_losses': val_losses}
        torch.save(state, os.path.join(self.save_path, 'best_checkpoint.pt'))

        self.val_loss_min = val_loss