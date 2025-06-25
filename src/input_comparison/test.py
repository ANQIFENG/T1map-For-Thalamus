#!/usr/bin/env python

import argparse
import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
from dataloader import ThalamusDataloader
from unet3d import UnetL5


def calculate_true_positive_per_class_without_background(pred, target, num_classes=13):
    true_positive_rates = np.zeros(num_classes)
    for i in range(1, num_classes + 1):
        pred_i = (pred == i)
        target_i = (target == i)
        intersection = np.logical_and(pred_i, target_i).sum()
        union = target_i.sum()
        true_positive_rate = intersection / union if union > 0 else 1.
        true_positive_rates[i - 1] = true_positive_rate
    return true_positive_rates


def calculate_volume_weighted_true_positive_without_background(true_positive_rates, data_fn_prefix, class_weights_path):
    class_weights_df = pd.read_csv(class_weights_path)
    weights_row = class_weights_df[class_weights_df['file_name'] == data_fn_prefix]
    if not weights_row.empty:
        volume_weights = weights_row.iloc[0, 1:].values.astype(float)
        volume_weights = volume_weights[1:]  # Exclude background
        volume_weighted_true_positive_rate = np.sum(true_positive_rates * volume_weights) / np.sum(volume_weights)
        return volume_weighted_true_positive_rate
    else:
        raise ValueError(f'No volume weights found for {data_fn_prefix} in the CSV file.')


def pad_to_original_size(pred, original_shape):
    padding = []
    for pred_dim, orig_dim in zip(pred.shape, original_shape):
        total_pad = orig_dim - pred_dim
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        padding.append((pad_before, pad_after))
    padded_pred = np.pad(pred, padding, mode='constant', constant_values=0)
    return padded_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thalamus segmentation testing")
    parser.add_argument("--test_batch_size", type=int, required=True, help="Batch size for testing")
    parser.add_argument("--device", type=str, required=True, help="Device to use (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--num_inputs", type=int, required=True, help="Number of input channels for the model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the input data.")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing the label data.")
    parser.add_argument("--original_data_dir", type=str, required=True, help="Directory for original data (used for getting affine info).")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory for checkpoints.")
    parser.add_argument("--class_weights_path", type=str, required=True, help="Path to the CSV file containing class weights.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save test results.")
    args = parser.parse_args()

    # Define label names
    label_names = ['AN', 'CM', 'LD', 'LP', 'MD', 'PuA', 'Pul', 'VA', 'VLa', 'VLP', 'VPL', 'VPM', 'CL']

    # Setup directories
    checkpoint_dir = args.checkpoint_dir
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    qualitative_results_save_path = os.path.join(save_path, 'qualitative_results')
    if not os.path.exists(qualitative_results_save_path):
        os.makedirs(qualitative_results_save_path)

    device = torch.device(args.device)
    model = UnetL5(in_dim=args.num_inputs, out_dim=14, num_filters=4, output_activation=nn.Softmax(dim=1)).to(device)
    model.eval()
    model.require_grad = False

    fold_scores = []
    all_test_data_scores = []

    # Perform 8-fold cross validation
    for fold in range(8):
        fold_str = str(fold)
        print("Processing fold", fold_str)
        checkpoint_path = os.path.join(checkpoint_dir, fold_str, 'best_checkpoint.pt')
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        model = model.to(device)

        test_loaders = ThalamusDataloader(data_dir=args.data_dir,
                                          label_dir=args.label_dir,
                                          batch_size=args.test_batch_size,
                                          split=fold_str,
                                          division="test",
                                          augment=False,
                                          shuffle=False,
                                          num_workers=2)
        current_fold_scores = []
        for batch_idx, (data, target, data_fn, label_fn) in enumerate(test_loaders):
            data_fn_prefix = data_fn[0].split('_data')[0]
            label_fn_prefix = label_fn[0].split('_label')[0]
            assert data_fn_prefix == label_fn_prefix
            print("The testing data is", data_fn_prefix)

            # Model prediction
            data = data.type(torch.float32).to(device)
            target = target.type(torch.int32).to(device)
            pred = model(data)
            pred_labels = torch.argmax(pred, dim=1).type(torch.int32)
            target_arr = target.squeeze().detach().cpu().numpy()
            pred_arr = pred_labels.squeeze().detach().cpu().numpy()

            # Quantitative testing
            true_positive = calculate_true_positive_per_class_without_background(pred_arr, target_arr)
            volume_weighted_tp = calculate_volume_weighted_true_positive_without_background(
                true_positive, data_fn_prefix, args.class_weights_path)
            test_data_scores = [data_fn_prefix] + true_positive.tolist() + [volume_weighted_tp]
            all_test_data_scores.append(test_data_scores)
            current_fold_scores.append(test_data_scores[1:])

            # Qualitative testing
            subject_id, session_id = data_fn_prefix.split('_', 1)
            original_data_folder = os.path.join(args.original_data_dir, subject_id, session_id)
            t1_files = [f for f in os.listdir(original_data_folder) if 'T1' in f and f.endswith('_wmnorm.nii.gz')]
            if t1_files:
                t1_file_path = os.path.join(original_data_folder, t1_files[0])
                affine_matrix = nib.load(t1_file_path).affine
                original_shape = nib.load(t1_file_path).shape
            else:
                raise FileNotFoundError(f"No T1 file found for {subject_id} {session_id}.")
            pred_arr_padded = pad_to_original_size(pred_arr, original_shape)
            out_filename = os.path.join(qualitative_results_save_path, f"{data_fn_prefix}_model_{fold_str}_pred_padded.nii.gz")
            nib.save(nib.Nifti1Image(pred_arr_padded.astype(np.int32), affine=affine_matrix), out_filename)
            print("Saved qualitative result to:", out_filename)

        fold_mean = np.mean(current_fold_scores, axis=0)
        fold_std = np.std(current_fold_scores, axis=0)
        fold_scores.append([f'Fold_{fold_str}'] + [f'{mean:.2f} +/- {std:.2f}' for mean, std in zip(fold_mean, fold_std)])

    overall_mean = np.mean([score[1:-1] for score in all_test_data_scores], axis=0)
    overall_std = np.std([score[1:-1] for score in all_test_data_scores], axis=0)
    overall_volume_weighted_mean = np.mean([score[-1] for score in all_test_data_scores])
    overall_volume_weighted_std = np.std([score[-1] for score in all_test_data_scores])

    all_test_data_scores.append(
        ['Overall'] + [f'{mean:.4f} +/- {std:.4f}' for mean, std in zip(overall_mean, overall_std)] +
        [f'{overall_volume_weighted_mean:.4f} +/- {overall_volume_weighted_std:.4f}']
    )
    fold_scores.append(
        ['Overall'] + [f'{mean:.4f} +/- {std:.4f}' for mean, std in zip(overall_mean, overall_std)] +
        [f'{overall_volume_weighted_mean:.4f} +/- {overall_volume_weighted_std:.4f}']
    )

    # Save quantitative testing results
    test_data_df = pd.DataFrame(all_test_data_scores,
                                columns=['Subject ID'] + label_names + ['Volume_Weighted_Mean'])
    test_data_df.to_csv(os.path.join(save_path, 'test_data_true_positive_rates.csv'), index=False)
    fold_df = pd.DataFrame(fold_scores,
                           columns=['Fold'] + label_names + ['Volume_Weighted_Mean'])
    fold_df.to_csv(os.path.join(save_path, 'fold_true_positive_rates.csv'), index=False)
    print("Quantitative testing results saved to:", save_path)
    print("Qualitative testing results saved to:", qualitative_results_save_path)


