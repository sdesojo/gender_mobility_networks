"""
Matching necessary for Figure 2 and Figure 4.a
Performs nearest neighbor matching by N and k to assess gender differences in user-month-level network metrics.
This script runs analyses across: all users, activity groups, and countries.

Input:
    - User-month-level network metrics.

Output:
    - Matched user sets and comparative statistics for gender-based network metric differences.

"""

import os
import pandas as pd
import pickle
import numpy as np
import tqdm
import itertools
import warnings

warnings.filterwarnings("ignore")

from utils.utils import GENDER, CTRY
from nearest_neighbor_matching.nnmatch_Nk_gender import (
    knn_matching_n_nu_mf2fm as nnmatch,
)


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
fname_metrics = "user_metrics_all_rsC.csv"
df_rs_c = pd.read_csv(input_path + fname_metrics)


# --- Define variables to run ---
CONTROL_VARS = ["visits", "locations"]
MAIN_VARS = [
    "density",
    "avg_unw_cluster",
    "n_cycles",
    "avg_len_cycles_all",
    "n_cycles_per_edge",
    "top1_degree_centrality",
    "top2_degree_centrality",
    "top3_degree_centrality",
    "nwide_MFPT_lovasz",
    "home_GMFPT_lovasz",
    "global_efficiency_edges",
    "global_efficiency_distance",
]

COLS = (
    ["useruuid", "start_month"]
    + [
        "gender",
        "GID_0",
        "activity_repertoire_groups",
    ]  # Empstatus, not used for matching but useful for analysis
    + MAIN_VARS
    + CONTROL_VARS
)

TEST = ["inactive", "moderate", "active", "all"]

# --- Define analysis to run ---
CASES_TO_RUN = ["countries", "activity_groups"]
n_iterations = 1000
max_delta_nu = {
    "all": 1,
    "inactive": 1,
    "moderate": 2,
    "active": 4,
}  # representing a 10% of median nu in group

print("> Start nnmatching")
print("> apply to: ", MAIN_VARS)
print("> control by: ", CONTROL_VARS)
print("> n_iterations:", n_iterations)

print("---------------------------------------------------------------------")


# ---------------------------------------------------------------------
# START ANALYSIS
# ---------------------------------------------------------------------

D_METRICS = [d + m for m, d in itertools.product(MAIN_VARS, ["abs_dif_", "rel_dif_"])]

CTRY_TO_RUN = ["all_ctry"]  # , 'all_ctry_wequalC']
if "countries" in CASES_TO_RUN:
    CTRY_TO_RUN += CTRY

for ctry in CTRY_TO_RUN:
    if ctry == "all_ctry":
        df_ = df[COLS].reset_index(drop=True)

    elif ctry == "all_ctry_wequalC":
        if get_rsC:
            df_ = df_rs_c[COLS].reset_index(
                drop=True
            )  ## Attention: Here the resampling is done once not once per sample.
        else:
            continue  # Skip if no resampled data available
    else:
        df_ = df[df["GID_0"] == ctry][COLS].reset_index(drop=True)

    d_res = {k: {"True": {}, "Shuffled": {}} for k in TEST}

    # Get all groups results for each country group
    for group in TEST:
        print(f">>> Running for activity group: {group}")
        if group != "all":
            df_group = df_[df_["activity_repertoire_groups"] == group].reset_index(
                drop=True
            )
        else:
            df_group = df_.copy()

        # Run True pairs KNN matching for the group
        df_match_true = nnmatch(
            df_group,
            MAIN_VARS,
            shuffled_gender=False,
            min_nnu=3,
        )

        # Drop pairs with k,N delta above threshold
        df_match_true = df_match_true[
            df_match_true["abs_abs_dif_" + "locations"] <= max_delta_nu[group]
        ]

        # Store results for the group
        d_res[group]["True"] = {
            dmlab: {
                "mean": np.mean(df_match_true[dmlab]),
                "mean_pos0": np.mean(df_match_true[df_match_true[dmlab] >= 0][dmlab]),
                "mean_neg0": np.mean(df_match_true[df_match_true[dmlab] <= 0][dmlab]),
                "ct_pos": df_match_true[df_match_true[dmlab] > 0].shape[0],
                "ct_neg": df_match_true[df_match_true[dmlab] < 0].shape[0],
                "ct_0": df_match_true[df_match_true[dmlab] == 0].shape[0],
            }
            for dmlab in D_METRICS
        }

        # Run Shuffled pairs KNN matching for the group across mutliple iterations
        RES_IT = []
        for it in tqdm.tqdm(range(n_iterations)):
            df_match_shuf = nnmatch(
                df_group,
                MAIN_VARS,
                shuffled_gender=True,
                min_nnu=3,
            )

            # Drop pairs with k,N delta above threshold
            df_match_shuf = df_match_shuf[
                df_match_shuf["abs_abs_dif_" + "locations"] <= max_delta_nu[group]
            ]

            # Save results for the group-iteration
            RES_IT.append(
                {
                    dmlab: {
                        "mean": np.mean(df_match_shuf[dmlab]),
                        "mean_pos0": np.mean(
                            df_match_shuf[df_match_shuf[dmlab] >= 0][dmlab]
                        ),
                        "mean_neg0": np.mean(
                            df_match_shuf[df_match_shuf[dmlab] <= 0][dmlab]
                        ),
                        "ct_pos": df_match_shuf[df_match_shuf[dmlab] > 0].shape[0],
                        "ct_neg": df_match_shuf[df_match_shuf[dmlab] < 0].shape[0],
                        "ct_0": df_match_shuf[df_match_shuf[dmlab] == 0].shape[0],
                    }
                    for dmlab in D_METRICS
                }
            )

        d_res[group]["Shuffled"] = RES_IT

    # Save results by country case
    fname = f"{ctry}_res_knn_match.pkl"
    with open(output_path + fname, "wb") as f:
        pickle.dump(d_res, f)
    print(f">> saved results for {ctry}")

print(">> saved all")
