#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_gradients(subjects_root, output_dir):
    """
    Aggregates dropout gradient CSV files from all subject folders and computes the overall mean gradients.

    """
    # Define class and modality labels
    class_labels = ['BG', 'AN', 'CM', 'LD', 'LP', 'MD', 'PuA', 'Pul', 'VA', 'VLa', 'VLP', 'VPL', 'VPM', 'CL']
    modality_titles = ['MPRAGE', 'FGATIR'] + sorted([f'SynT1_{TI}' for TI in range(400, 1401, 20)])

    # List to store gradients from all subjects and iterations
    all_gradients = []

    # Iterate through each subject folder
    subject_folders = [
        os.path.join(subjects_root, subj)
        for subj in os.listdir(subjects_root)
        if os.path.isdir(os.path.join(subjects_root, subj))
    ]

    for subject_folder in subject_folders:
        print(f"Processing subject: {os.path.basename(subject_folder)}")
        # Get all dropout CSV files for the subject
        dropout_files = sorted([
            os.path.join(subject_folder, f)
            for f in os.listdir(subject_folder)
            if f.startswith('dropout_') and f.endswith('.csv')
        ])

        # Load and store each CSV file's data
        for file in dropout_files:
            data = pd.read_csv(file, index_col=0)
            all_gradients.append(data.values)

    # Convert list to numpy array: shape (total_iterations, num_modalities, num_classes)
    all_gradients = np.array(all_gradients)

    # Compute the mean gradient across all iterations and subjects (axis=0)
    mean_gradients = np.mean(all_gradients, axis=0)

    # Save the mean gradients to a CSV file
    mean_df = pd.DataFrame(mean_gradients, columns=class_labels, index=modality_titles)
    mean_csv_path = os.path.join(output_dir, 'overall_gradient_mean.csv')
    mean_df.to_csv(mean_csv_path)
    print(f"Saved overall mean gradients to {mean_csv_path}")

    return mean_df


def plot_ois(mean_df, output_dir):
    """
    Plots a horizontal bar plot of the Overall Importance Scores (OIS) based on the aggregated mean gradients.

    """
    # Calculate the average contribution score across all nuclei for each modality
    mean_df['AVG'] = mean_df.sum(axis=1)

    # Rename modalities: replace 'SynT1_' with 'TI_'
    mean_df.rename(index=lambda x: x.replace('SynT1_', 'TI_'), inplace=True)

    # Sort modalities by the average contribution score in descending order
    avg_scores = mean_df.sort_values(by='AVG', ascending=False)['AVG']

    # Create the horizontal bar plot
    plt.figure(figsize=(12, 3))
    sns.barplot(x=avg_scores.index, y=avg_scores, palette="RdYlGn")
    plt.title('Ranking of Images by Overall Importance Scores (OIS)', fontsize=10)
    plt.xlabel('Images', fontsize=10)
    plt.ylabel('Overall Importance Score', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, 'ranking_of_overall_important_scores.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved successfully at {output_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Aggregate dropout gradients and plot Overall Importance Scores (OIS).")
    parser.add_argument("--subjects_root", type=str, required=True, help="Path to the root folder containing subject subfolders with dropout CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where the aggregated CSV and plot will be saved.")
    args = parser.parse_args()

    # Create output directory if not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Aggregate gradients and compute overall mean gradients
    mean_df = aggregate_gradients(args.subjects_root, args.output_dir)

    # Plot the Overall Importance Scores (OIS)
    plot_ois(mean_df, args.output_dir)
