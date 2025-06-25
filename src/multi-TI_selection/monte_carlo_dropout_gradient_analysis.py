import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataloader import ThalamusDataloader
from unet3d_dropout import UnetL5WithDropout


# Class and Modality Labels
class_labels = ['BG', 'AN', 'CM', 'LD', 'LP', 'MD', 'PuA', 'Pul', 'VA', 'VLa', 'VLP', 'VPL', 'VPM', 'CL']
modality_titles = ['MPRAGE', 'FGATIR'] + sorted(
    [f'SynT1_{TI}' for TI in range(400, 1401, 20)])


# Enable Dropout during testing
def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout3d):
            module.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gradients Calculation for each subject with Monte Carlo Dropout")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--dropout_iterations", type=int, default=100, help="MC Dropout runs")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="MC Dropout rate")
    parser.add_argument("--num_classes", type=int, default=14, help="Number of classes")
    parser.add_argument("--num_channels", type=int, default=53, help="Number of channels")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--label_dir", type=str, required=True, help="Label directory")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="Checkpoint directory")
    parser.add_argument("--save_path", required=True, type=str, help="Save path")

    args = parser.parse_args()

    # Initialize the model
    model = UnetL5WithDropout(in_dim=args.num_channels, out_dim=args.num_classes, num_filters=4, dropout_rate=args.dropout_rate,
                              output_activation=nn.Softmax(dim=1)).to(args.device)
    model.eval()

    # Load each fold and test
    for split_idx in range(8):
        split_idx = str(split_idx)
        print(f'Processing fold {split_idx}...')

        # Load checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, split_idx, 'best_checkpoint.pt')
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=False)
        model.to(args.device)

        # Activate Dropout
        enable_dropout(model)

        # Prepare test data
        test_loader = ThalamusDataloader(data_dir=args.data_dir,
                                         label_dir=args.label_dir,
                                         batch_size=args.test_batch_size,
                                         split=split_idx,
                                         division="test",
                                         augment=False,
                                         shuffle=False)

        for batch_idx, (data, target, data_fn, label_fn) in enumerate(test_loader):
            # Identify the test subject
            subject_id = data_fn[0].split('_data')[0]
            print(f'Testing subject: {subject_id}')

            # Create a subfolder for the subject
            subject_folder = os.path.join(args.save_path, subject_id)
            os.makedirs(subject_folder, exist_ok=True)

            # Move data to device
            data = data.type(torch.float32).to(args.device)
            data.requires_grad = True

            # Perform multiple Dropout runs per subject
            for iteration in range(args.dropout_iterations):
                print(f'Dropout Iteration {iteration + 1}/{args.dropout_iterations}')

                # Forward pass
                pred = model(data)

                # Initialize storage for this iteration
                gradient_iteration = np.zeros((args.num_channels, args.num_classes), dtype=np.float32)

                # Compute gradients for each class
                for target_class in range(args.num_classes):
                    print(f"Computing gradients for class {target_class}")

                    # Compute class score
                    class_score = pred[:, target_class].sum()  # [batch, num_classes, H, W, D]

                    # Zero out any previously accumulated gradients
                    model.zero_grad()

                    # Backpropagation to compute gradients
                    class_score.backward(retain_graph=True)

                    # Get the gradients with respect to the input
                    gradients = data.grad.detach().cpu().numpy().squeeze()  # Shape: [53, H, W, D]

                    # Compute absolute sum for each channel (TI image)
                    absolute_sums = np.sum(np.abs(gradients), axis=(1, 2, 3))  # Shape: [53]

                    # Compute total sum for normalization
                    total_sum = absolute_sums.sum()

                    # Fill in the gradient_iteration table with normalization
                    gradient_iteration[:, target_class] = absolute_sums / total_sum

                # Save the gradients per MC run to a CSV file
                iteration_output_path = os.path.join(subject_folder, f"dropout_{iteration + 1:03d}.csv")
                gradient_df = pd.DataFrame(gradient_iteration, columns=class_labels, index=modality_titles)
                gradient_df.to_csv(iteration_output_path)
                print(f"Saved Dropout {iteration + 1} gradients to {iteration_output_path}")

            print(f"Finished processing subject: {subject_id}")
