#!/usr/bin/env python

import torch
import torch.nn as nn


def one_hot_encoding(gt, gt_values, num_classes):

    if gt.dim() == 5 and gt.size(1) == 1:
        gt = gt.squeeze(1)
    one_hot = torch.zeros(gt.size(0), num_classes, *gt.size()[1:], device=gt.device)
    for idx in range(num_classes):
        one_hot[:, idx, ...] = (gt == gt_values[idx]).float()
    return one_hot


class DiceLoss(nn.Module):

    def __init__(self, num_classes, eps=1e-6, isOneHot=False, gt_values=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.isOneHot = isOneHot
        self.gt_values = gt_values

    def forward(self, prediction, ground_truth):

        if not self.isOneHot:
            if self.gt_values is None:
                raise ValueError("gt_values must be provided if ground truth is not one-hot encoded.")
            ground_truth = one_hot_encoding(ground_truth, self.gt_values, self.num_classes)

        intersection = torch.sum(prediction * ground_truth, dim=(2, 3, 4))
        union = torch.sum(prediction ** 2, dim=(2, 3, 4)) + torch.sum(ground_truth ** 2, dim=(2, 3, 4))

        dice_score_channel = (2 * intersection + self.eps) / (union + self.eps)
        dice_score_channel = torch.mean(dice_score_channel, dim=0)

        dice_score_overall = torch.mean(dice_score_channel)
        dice_loss_overall = 1 - dice_score_overall

        return dice_loss_overall