#!/usr/bin/env python

import os
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if "Overall" in df.iloc[-1, 0]:
        df = df.iloc[:-1, :]
    return df


ref_csv = r"../exp5/test_data_true_positive_rates.csv"  # T1-map only experiment
ref_df = load_data(ref_csv)

out_dir = "../wilcoxon_test"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "overall_vs_exp5.csv")

columns_to_compare = [
    "AN", "CM", "LD", "LP", "MD", "PuA", "Pul", "VA",
    "VLa", "VLP", "VPL", "VPM", "CL", "Volume_Weighted_Mean"]

candidate_exps = ['exp1', 'exp2', 'exp3', 'exp4', 'exp6', 'exp7', 'exp8', 'exp9']  # other configurations

all_results = []
for exp in candidate_exps:
    candidate_csv = f"../{exp}/test/test_data_true_positive_rates.csv"
    candidate_df = load_data(candidate_csv)

    for col in columns_to_compare:
        candidate_values = candidate_df[col].astype(float).values
        ref_values = ref_df[col].astype(float).values

        stat, p_value = wilcoxon(candidate_values, ref_values)
        all_results.append((exp, col, stat, p_value))


results_df = pd.DataFrame(all_results, columns=["Experiment", "Column", "Stat", "P_value"])
p_values = results_df["P_value"].values
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
results_df["P_value_corrected"] = pvals_corrected
results_df["Reject"] = reject
results_df.to_csv(out_path, index=False, float_format="%.6g")

