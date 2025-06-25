#!/usr/bin/env python

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from early_stopping import EarlyStopping
from loss import DiceLoss
from dataloader import ThalamusDataloader
from unet3d import UnetL5

device = torch.device("cuda")
gt_values = list(range(14))

# fix random seeds
global_seed = 42
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(global_seed)


# train
def train(train_loaders, model, lossfn, optimizer, epoch):

    model.train()
    progress_bar = tqdm(train_loaders, desc="Training")

    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(progress_bar):

        data = data.type(torch.float32).to(device)
        target = target.type(torch.float32).to(device)

        optimizer.zero_grad()
        pred = model(data)

        loss = lossfn(pred, target)
        loss.backward()

        optimizer.step()
        total_loss += loss.data.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_description('epoch index {} loss:{:.6f}'.format(epoch,  avg_loss))

    return avg_loss


# validation
def val(val_loaders, model, lossfn, epoch):

    model.eval()
    progress_bar = tqdm(val_loaders, desc='Validation')

    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):

            data = data.type(torch.float32).to(device)
            target = target.type(torch.float32).to(device)

            pred = model(data)

            loss = lossfn(pred, target)

            total_loss += loss.data.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_description('Epoch: {} test loss: {:.6f}'.format(epoch, avg_loss))

    return avg_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train 3d unet using Dice Loss.")
    parser.add_argument('data_dir', type=str, help='Root folder for the data.')
    parser.add_argument('label_dir', type=str, help='Root folder for the ground truth labels.')
    parser.add_argument('save_path', type=str, help='Folder to save the training results.')
    parser.add_argument('--split', type=str, required=True, help='Data split to use.')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation.')
    parser.add_argument('--resume', action='store_true', help='Whether to resume training from the last Best Epoch.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay.')
    parser.add_argument('--num_filters', type=int, default=4, help='The number of initial filters.')
    parser.add_argument('--num_inputs', type=int, required=True, help='The number of input channels.')
    parser.add_argument('--num_outputs', type=int, default=14, help='The number of output classes.')
    args = parser.parse_args()

    print("=="*50)
    print("Eight Fold Experiment Split ", args.split)
    print("=="*50)

    # creating folders to save results
    if not os.path.exists(args.save_path):
        print('creating:', args.save_path)
        os.makedirs(args.save_path)

    sub_path = os.path.join(args.save_path, args.split)
    if not os.path.exists(sub_path):
        print('creating:', sub_path)
        os.mkdir(sub_path)

    # model, loss, optimizer, scheduler and early stop
    model = UnetL5(in_dim=args.num_inputs, out_dim=args.num_outputs, num_filters=args.num_filters, output_activation=nn.Softmax(dim=1)).to(device)
    lossfn = DiceLoss(num_classes=args.num_outputs, isOneHot=False, gt_values=gt_values)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9)
    early_stopping = EarlyStopping(patience=15, verbose=True, save_path=sub_path)

    # whether resume training
    startEpoch = -1
    if args.resume:
        print("Resume Training from the last best checkpoint.")
        resumed_checkpoint = os.path.join(sub_path, 'best_checkpoint.pt')
        if os.path.exists(resumed_checkpoint):
            resume_state = torch.load(resumed_checkpoint, map_location=torch.device('cuda'))
            model.load_state_dict(resume_state['state_dict'])
            startEpoch = resume_state['epoch']
            train_losses = resume_state['train_losses']
            val_losses = resume_state['test_losses']
            optimizer.load_state_dict(resume_state['optimizer_state_dict'])
            scheduler.load_state_dict(resume_state['scheduler_state_dict'])
        else:
            print("No checkpoint found, starting from scratch.")
            train_losses, val_losses = [], []
    else:
        train_losses, val_losses = [], []

    # start training
    for epoch in range(startEpoch+1, args.epochs):

        train_loaders = ThalamusDataloader(data_dir=args.data_dir,
                                           label_dir=args.label_dir,
                                           batch_size=args.train_batch_size,
                                           split=args.split,
                                           division="train",
                                           augment=True,
                                           shuffle=True,
                                           num_workers=2
                                           )

        val_loaders = ThalamusDataloader(data_dir=args.data_dir,
                                         label_dir=args.label_dir,
                                         batch_size=args.val_batch_size,
                                         split=args.split,
                                         division="val",
                                         augment=False,
                                         shuffle=False,
                                         num_workers=2
                                         )

        # train model
        train_avg_loss = train(train_loaders, model, lossfn, optimizer, epoch)
        train_losses.append(train_avg_loss)

        # test model
        val_avg_loss = val(val_loaders, model, lossfn, epoch)
        val_losses.append(val_avg_loss)

        # adjust lr
        scheduler.step(val_avg_loss)

        # save train loss and test loss for each epoch
        np.savez(os.path.join(sub_path, 'train_losses.npz'), train_losses, allow_pickle=True)
        np.savez(os.path.join(sub_path, 'val_losses.npz'), val_losses, allow_pickle=True)

        # plot training and testing loss
        fig = plt.figure(figsize=(20, 24))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        ax1.plot(train_losses, color='darkred', marker='o')
        ax1.set_title('train loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')

        ax2.plot(val_losses, color='b', marker='o')
        ax2.set_title('val loss')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')

        ax3.plot(train_losses, color='darkred', marker='o')
        ax3.plot(val_losses, color='b', marker='o')
        ax3.set_title('training and validation loss')
        ax3.set_xlabel('epoch')
        ax3.set_ylabel('loss')
        ax3.legend(['train', 'val'])
        plt.tight_layout()
        plt.savefig(os.path.join(sub_path, 'losses.png'))

        # early stop
        early_stopping(val_avg_loss, model, train_losses, val_losses, optimizer, scheduler, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered at epoch:", epoch)
            break
