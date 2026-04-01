"""
Script to analyze gender differences in tour efficiency across countries and samples.
This anlaysis will be visualized in figure 4 of the paper.

Analysis includes:
- Compute gender difference in tour efficiency across reward bins and by quantiles of activity (number of visits).

Original computation in:
    tc_journeys/data/01_src_gengap_lenght_eff.py
    tc_journeys/data/02_eff_gengap.py

"""

import os
import pandas as pd
import pickle
import numpy as np

from utils.utils import GENDER, CTRY, perc2quant
from statistical_analysis.basic_stats import apply_bootstrap

import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# DEFINE INPUT DATA, OUTPUT PATH, AND ANALYSIS TO RUN
# ---------------------------------------------------------------------

# --- Input / Output Paths ---
input_path = ""
output_path = ""

# --- Load ---
fname_metrics = "user_metrics_all.csv"
df = pd.read_csv(input_path + fname_metrics)

get_rsC = True
if get_rsC:
    fname_metrics = "user_metrics_all_rsC.csv"
    df_rs_c = pd.read_csv(input_path + fname_metrics)


# --- Define variables to run ---
cost_met = "total_cost_travel_distance"  #'cost_logsumkm'
reward_met = "total_reward_home_distance"  #'reward_home_logsumkm'
eff_met = "tour_efficiency_dist"  #'seq_efficency_home_logsumkm'
quant_var = "visits"

u_cols = ["useruuid", "gender", "GID_0", "start_month"]

# RUN_CTRY = ['all_ctry', 'all_ctry_wequalC'] #+ CTRY
# RUN_CTRY = CTRY[:]
RUN_CTRY = ["USA"]

# --- Define analysis to run ---
run_geneff_rewbins = True


print(f"> Input data: {input_path + fname_metrics}")
print(f"> Output path: {output_path}")

print(f"> Run run_geneff_rewbins: {run_geneff_rewbins}")

print("---------------------------------------------------------------------")


# ---------------------------------------------------------------------
# START ANALYSIS
# ---------------------------------------------------------------------

## --- Historgrams of Nk distributions ---
if run_geneff_rewbins:

    # Define nsamples per bs iteration
    nsamples = 1000

    # dic_ks = {ctry: {} for ctry in ['all_ctry', 'all_ctry_wequalC'] + CTRY }
    for ctry in RUN_CTRY:
        print(f">>> Running for country: {ctry} <<<")
        if ctry == "all_ctry":
            df_ = df.copy()
        elif ctry == "all_ctry_wequalC":
            if get_rsC:
                df_ = (
                    df_rs_c.copy()
                )  ## Attention: Here the resampling is done once not once per sample.
            else:
                continue  # Skip if no resampled data available
        else:
            df_ = df[df["GID_0"] == ctry].copy()

        # Select relevant columns
        df_mseq = df_[u_cols + [cost_met, reward_met, eff_met, quant_var]].dropna(
            subset=[cost_met, reward_met, eff_met]
        )
        df_mseq = df_mseq[
            df_mseq[eff_met] >= 0
        ]  # only consider users with 0, or postive eff (negative cases are minimal, and due to noise on etsimation dist travel)

        # Get log of reward and cost for better binning and visualization
        df_mseq[reward_met + "_log"] = np.log10(df_mseq[reward_met])
        df_mseq[cost_met + "_log"] = np.log10(df_mseq[cost_met])

        # Define reward bins
        dist = df_mseq[df_mseq[reward_met + "_log"] > 0][reward_met + "_log"]
        nbins = 30
        MINv = np.log10(0.2 * 30)  # 0.2km min journey length per day
        MAXv = np.log10(300 * 30)  # avg. 300km per day (95th percentile?)

        print(10 ** dist.describe())
        print(
            f">> {reward_met+ '_log'} |  5th perc.: ",
            round(10 ** (np.percentile(dist, 5)), 2),
            "applied capp is avg0.2/day: ",
            0.2 * 30,
        )
        print(
            f">> {reward_met+ '_log'} |  95th perc.: ",
            round(10 ** (np.percentile(dist, 95)), 2),
            "applied capp is avg300/day: ",
            300 * 30,
        )

        REWARD_BINS = np.linspace(MINv, MAXv, num=nbins)
        df_mseq["reward_bin"] = pd.cut(
            df_mseq[reward_met + "_log"], bins=REWARD_BINS, include_lowest=True
        )

        # Define quantiles of activity
        df_mseq["q_bin"] = pd.qcut(
            df_mseq[quant_var], q=5, labels=False, duplicates="drop"
        )

        # Bootstrap efficiency and cost for each reward bin and quantiles of activity
        df_bs_ef = (
            df_mseq.groupby(["gender", "reward_bin", "q_bin"])
            .apply(apply_bootstrap, var=eff_met, n_samples=nsamples)
            .reset_index()
        )

        df_bs_cost = (
            df_mseq.groupby(["gender", "reward_bin", "q_bin"])
            .apply(apply_bootstrap, var=cost_met, n_samples=nsamples)
            .reset_index()
        )
        print(
            ">> Bootstrapping done for efficiency", df_bs_ef.columns.tolist()
        )  # ['gender', 'reward_bin', 'q_bin', 'median', 'standard_error']

        # Count users per bin
        df_ct = (
            df_mseq.groupby(["gender", "reward_bin", "q_bin"])[["useruuid"]]
            .count()
            .rename(columns={"useruuid": "n_user_months"})
            .reset_index()
        )
        df_bs_cost = df_bs_cost.merge(df_ct, on=["gender", "reward_bin", "q_bin"])
        df_bs_ef = df_bs_ef.merge(df_ct, on=["gender", "reward_bin", "q_bin"])

        # Save bs results
        df_bs_ef.to_csv(
            os.path.join(output_path, f"fig4_{ctry}_bs_eff_by_totrewbins_Nquant.csv"),
            index=False,
        )
        print(">> Saved bs for efficiency")

        df_bs_cost.to_csv(
            os.path.join(output_path, f"fig4_{ctry}_bs_cost_by_totrewbins_Nquant.csv"),
            index=False,
        )
        print(">> Saved bs for cost")

        ## > Get difference by gender in efficiency
        for met, df_bs in zip(["eff", "cost"], [df_bs_ef, df_bs_cost]):

            df_M = df_bs[df_bs["gender"] == "MALE"].drop("gender", axis=1)
            df_F = df_bs[df_bs["gender"] == "FEMALE"].drop("gender", axis=1)
            df_result = df_M.merge(
                df_F, on=["reward_bin", "q_bin"], suffixes=("_M", "_F")
            )

            # Keep only bins with at least 200 M and F users
            min_u = 170
            df_result = df_result[
                (df_result["n_user_months_M"] >= min_u)
                & (df_result["n_user_months_F"] >= min_u)
            ]
            df_result["xbins_med"] = df_result["reward_bin"].apply(
                lambda x: np.median([x.left, x.right])
            )

            # Mean
            df_result["bs_median_reldiffsym"] = (
                2
                * (df_result["median_M"] - df_result["median_F"])
                / (df_result["median_M"] + df_result["median_F"])
            )

            # Standard error
            df_result["bs_se_reldiffsym"] = (
                2 / (df_result["median_M"] + df_result["median_F"])
            ) * np.sqrt(
                df_result["standard_error_M"] ** 2 + df_result["standard_error_F"] ** 2
            )

            # Save as
            df_result.to_csv(
                os.path.join(
                    output_path,
                    f"fig4_{ctry}_bs_gendergap_{met}_by_totrewbins_Nquant.csv",
                ),
                index=False,
            )
            print(
                ">> Saved bs gender gap for efficiency",
                f"fig4_{ctry}_bs_gendergap_{met}_by_totrewbins_Nquant.csv",
            )
