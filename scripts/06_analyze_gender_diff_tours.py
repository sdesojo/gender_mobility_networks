"""
Anlaysis necessary for Figure 3
"""

### OPEN TOURS
### BOTSTRAO BASED ON EACH EXECRISE
### PALINDROMES

# df_tours = df_tours[df_tours['tour_total_distance'] > 0].copy()


import os
import pandas as pd
import pickle
import numpy as np
import tqdm
import itertools
import re
import warnings

warnings.filterwarnings("ignore")

from utils.utils import GENDER, CTRY, pooled_se

# ---------------------------------------------------------------------
# DEFINE INPUT DATA, OUTPUT PATH, AND ANALYSIS TO RUN
# ---------------------------------------------------------------------


# --- Input Data ---
input_path = ""
fname_metrics = "user_tours_all.csv"

# --- Define Output Dir ---
output_path = ""


# --- Define parameters ---
n_iterations = 1000
min_u = 200

leng_met = "tour_total_stops"
leng_bins = "leng_bin"

LENG_LABELS = ["3\nABA", "4\nABCA", "5\nABCDA", "6"] + ["7-10", "11-20", "20+"]
dLENG = {
    "3\nABA": "Back-and-forth",
    "4\nABCA": "2-stop",
    "5\nABCDA": "3-stop",
    "6": "4-stop",
    "7-10": "7–10-stop",
    "11-20": "11–20-stop",
    "20+": ">20-stop",
}


# -- Define anlaysis to run ---
run_palindromes = False
run_tours_length = False
run_tours_gendergap_leng = True
run_tours_gendergap_leng_dist = True
run_tours_gendergap_leng_duration = True

print("> Start analysis of gender differences in tours")
print("> run_palindromes:", run_palindromes)
print("> run_tours_length:", run_tours_length)
print("> run_tours_gendergap_leng:", run_tours_gendergap_leng)
print("> run_tours_gendergap_leng_dist:", run_tours_gendergap_leng_dist)
print("> run_tours_gendergap_leng_duration:", run_tours_gendergap_leng_duration)

# ---------------------------------------------------------------------
# START ANALYSIS
# ---------------------------------------------------------------------

df_tours = pd.read_csv(input_path + fname_metrics)

# -- PALINDROMES ---------
if run_palindromes:

    def parse_stop_sequence(s):
        return re.findall(r"'([^']+)'", s)

    def is_palindrome(seq):
        """Check if a sequence is a palindrome."""
        return int(seq == seq[::-1])

    df_p = df_tours.dropna(subset=["tour", leng_met])
    df_p["tour_clean"] = df_p["tour"].apply(parse_stop_sequence)
    df_p["is_palindrome"] = df_p["tour_clean"].apply(lambda x: is_palindrome(x))

    # Step 0: Define bootstrap parameters
    dic_palind = {}
    for min_len in [3, 5]:
        # df_jrny_filt = df_p[df_p[leng_met]>= min_len]
        # df_res = df_jrny_filt.groupby(['gender']).agg({'is_palindrome': ['sum','count']})
        # df_res['%_palindrome'] = 100* df_res[('is_palindrome', 'sum')]/df_res[('is_palindrome', 'count')]
        # dic_palind[min_len] = df_res

        bootstrap_samples = []
        for _ in range(n_iterations):
            # Resample with replacement
            df_resampled = df_p.sample(frac=1, replace=True)

            # Step 1: Filter journeys by minimum length
            df_jrny_filt = df_resampled[df_resampled[leng_met] >= min_len]

            # Step 2: Compute palindrome statistics
            df_res = df_jrny_filt.groupby(["gender"]).agg(
                {"is_palindrome": ["sum", "count"]}
            )
            df_res["%_palindrome"] = (
                100
                * df_res[("is_palindrome", "sum")]
                / df_res[("is_palindrome", "count")]
            )
            bootstrap_samples.append(df_res.reset_index()[["gender", "%_palindrome"]])

        # Step 3: Compute bootstrap statistics
        df_bs = pd.concat(bootstrap_samples)
        df_bs.columns = df_bs.columns.droplevel(1)
        df_bs_summary = (
            df_bs.groupby(["gender"])
            .agg(
                mean_palind=("%_palindrome", "mean"),
                se_palind=(
                    "%_palindrome",
                    "std",
                ),  # lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
            )
            .reset_index()
        )
        dic_palind[min_len] = df_bs_summary

    # Step 4: Save results
    save_to = output_path + f"dic_palind_BS{n_iterations}.pkl"
    with open(save_to, "wb") as f:
        pickle.dump(dic_palind, f)
    print(">> saved: run_palindromes")


# -- SHARE OF JOURNEYS PER JOURNEY LENGHT ---------
if run_tours_length:

    df_tours_ = df_tours.dropna(subset=["tour", leng_met])
    df_frac = (
        df_tours.groupby(leng_bins)
        .agg({"useruuid": "count"})
        .reset_index()
        .rename(columns={"useruuid": "tot"})
    )
    df_frac["frac"] = df_frac["tot"] / df_frac["tot"].sum()

    df_frac["leng_desc"] = df_frac["leng_bin"].map(dLENG)
    df_frac["leng_bin"] = pd.Categorical(
        df_frac["leng_bin"], categories=LENG_LABELS, ordered=True
    )
    df_frac = df_frac.sort_values(by="leng_bin")
    df_frac = df_frac[["leng_bin", "leng_desc", "frac"]]

    save_to = output_path + f"frac_tours_length.pkl"
    with open(save_to, "wb") as f:
        pickle.dump(df_frac, f)
    print(">> saved: run_tours_length")


# -- GENDER GAP BY JOURNEY LENGHT, ACTIVITY LEVEL ---------
if run_tours_gendergap_leng:

    dic_test = {}
    dic_test_ctry = {}

    for test in ["all", "moderate", "inactive", "active"]:
        if test == "all":
            df_jrny = df_tours.dropna(subset=["tour", leng_met])
        elif test == "moderate":
            df_jrny = df_tours[
                df_tours["activity_repertoire_groups"] == "moderate"
            ].dropna(subset=["tour", leng_met])
        elif test == "inactive":
            df_jrny = df_tours[
                df_tours["activity_repertoire_groups"] == "inactive"
            ].dropna(subset=["tour", leng_met])
        elif test == "active":
            df_jrny = df_tours[
                df_tours["activity_repertoire_groups"] == "active"
            ].dropna(subset=["tour", leng_met])

        bootstrap_samples = []
        for _ in range(n_iterations):

            df_resampled = df_jrny.sample(frac=1, replace=True)

            # Step 1: Get count of women/men per journey length bin, country
            GROUP = ["gender", "leng_bin", "GID_0"]
            df_g = df_resampled.groupby(GROUP).agg({leng_met: "count"}).reset_index()
            df_g = df_g.pivot(
                index=["leng_bin", "GID_0"], columns=["gender"], values=leng_met
            ).reset_index()

            # Step 2: Normalize by the total volume of jrneys per gender, country and activity group
            levels = ["MALE", "FEMALE"]
            for gen in levels:
                lab = gen[0] + "norm"
                df_g[lab] = df_g.groupby("GID_0")[gen].transform(lambda x: x / x.sum())

            df_g["gap"] = df_g["Mnorm"] - df_g["Fnorm"]
            df_g["ct_users"] = df_g["MALE"] + df_g["FEMALE"]
            df_g["frac_users"] = df_g.groupby(["leng_bin", "GID_0"])[
                "ct_users"
            ].transform(lambda x: x / x.sum())

            ## Step 3: Clean lenghts with not enough users
            df_g["gap_clean"] = df_g.apply(
                lambda row: (
                    np.nan
                    if (row["MALE"] < min_u) or (row["FEMALE"] < min_u)
                    else row["gap"]
                ),
                axis=1,
            )

            # Append the result to the list
            bootstrap_samples.append(
                df_g[["leng_bin", "GID_0", "frac_users", "gap_clean"]].copy()
            )

        # Step 4: Create a DataFrame from the bootstrap samples
        df_bs = pd.concat(bootstrap_samples, ignore_index=True)

        # Step 5: Calculate the mean and standard error for each bin
        df_bs_summary = (
            df_bs.groupby(["leng_bin", "GID_0"])
            .agg(
                mean_gap=("gap_clean", "mean"),
                mean_frac=("frac_users", "mean"),
                se_gap=(
                    "gap_clean",
                    "std",
                ),  # lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                se_frac=("frac_users", "std"),
            )
            .reset_index()
        )

        df_c = df_bs_summary[df_bs_summary.columns]
        df_c["leng_desc"] = df_c["leng_bin"].map(dLENG)
        df_c["leng_bin"] = pd.Categorical(
            df_c["leng_bin"], categories=LENG_LABELS, ordered=True
        )
        df_c = df_c.sort_values(by="leng_bin")

        dic_test_ctry[test] = df_c

        # Step 6: Get pooled standard error across countries for each bin
        df_a = (
            df_bs_summary.groupby("leng_bin")
            .agg({"mean_gap": "mean", "se_gap": pooled_se})
            .reset_index()
        )
        df_a["leng_desc"] = df_a["leng_bin"].map(dLENG)
        df_a["leng_bin"] = pd.Categorical(
            df_a["leng_bin"], categories=LENG_LABELS, ordered=True
        )
        df_a = df_a.sort_values(by="leng_bin")

        dic_test[test] = df_a

    # Step 7: Save results
    save_to = (
        output_path + f"gendergap_tours_length_byctry_activity_BS{n_iterations}.pkl"
    )
    with open(save_to, "wb") as f:
        pickle.dump(dic_test_ctry, f)
    print(f">> saved: run_tours_gendergap_leng at {save_to}")

    save_to = (
        output_path + f"gendergap_tours_length_pooled_activity_BS{n_iterations}.pkl"
    )
    with open(save_to, "wb") as f:
        pickle.dump(dic_test, f)
    print(f">> saved: run_tours_gendergap_leng at {save_to}")


# -- GENDER GAP BY JOURNEY LENGHT, ACTIVITY LEVEL ACROSS DISTANCE BINS ---------
if run_tours_gendergap_leng_dist:

    # Step 0: Define vars
    reward_met = "tour_max_reward"
    nbins = 15
    capped_dist = True
    min_val = 0.1
    max_val = 300

    dic_test = {}
    dic_test_ctry = {}
    for test in ["all", "moderate", "inactive", "active"]:
        if test == "all":
            df_jrny = df_tours.dropna(subset=["tour", leng_met])
        elif test == "moderate":
            df_jrny = df_tours[
                df_tours["activity_repertoire_groups"] == "moderate"
            ].dropna(subset=["tour", leng_met])
        elif test == "inactive":
            df_jrny = df_tours[
                df_tours["activity_repertoire_groups"] == "inactive"
            ].dropna(subset=["tour", leng_met])
        elif test == "active":
            df_jrny = df_tours[
                df_tours["activity_repertoire_groups"] == "active"
            ].dropna(subset=["tour", leng_met])

        # for reward_met in ['jrny_max_reward', 'jrny_tot_reward']:
        # print('>> start: eval_len_by_dist_bs for ', reward_met, 'capped:'  , capped_dist)
        # Step BS-1: Define bootstrap parameters
        bootstrap_samples = []

        for _ in range(n_iterations):

            # Step 0: Resample with replacement
            df_jrny_ = df_jrny.sample(frac=1, replace=True)
            df_b = df_jrny_[["gender", "GID_0", leng_met, reward_met, "leng_bin"]]

            # Step 1: convert dist to km
            df_b[reward_met] = df_b[reward_met].apply(lambda x: x / 1000)

            # Step 2: Define distance bins
            dist = df_b[df_b[reward_met] >= min_val][reward_met]

            if capped_dist:
                DIST_BINS = np.logspace(np.log10(min_val), np.log10(max_val), num=nbins)
                df_b["dist_bin"] = pd.cut(
                    df_b[reward_met], bins=DIST_BINS, include_lowest=True
                )

            else:
                MAXv = dist.max()
                DIST_BINS = np.logspace(np.log10(min_val), np.log10(MAXv), num=nbins)
                df_b["dist_bin"] = pd.cut(
                    df_b[reward_met], bins=DIST_BINS, include_lowest=True
                )

            df_b = df_b.sort_values(["dist_bin", "leng_bin"])

            # Step 3: Get count of women/men per defined-bin
            df_g = (
                df_b.groupby(["leng_bin", "dist_bin", "gender", "GID_0"])
                .agg({leng_met: "count"})
                .reset_index()
            )
            df_g = df_g.pivot(
                index=["dist_bin", "leng_bin", "GID_0"],
                columns="gender",
                values=leng_met,
            ).reset_index()

            ## Step 4: Normalize by the total volume of users per gender FOR EACH DIST BIN.
            levels = ["MALE", "FEMALE"]
            for gen in levels:
                lab = gen[0] + "norm"
                df_g[lab] = df_g.groupby(["dist_bin", "GID_0"])[gen].transform(
                    lambda x: x / x.sum()
                )
                df_g[lab] = df_g[lab].fillna(0)

            # Step 5: Get difference in %M and %F per bin
            df_g["gap"] = df_g["Mnorm"] - df_g["Fnorm"]
            # df_g['gap_sym'] = 2*(df_g['Mnorm'] - df_g['Fnorm']) / (df_g['Mnorm'] + df_g['Fnorm'])

            # Step 6: Get Fraction Users per distance bin for each len bin separetley
            df_g["ct_users"] = df_g["MALE"] + df_g["FEMALE"]
            df_g["frac_users"] = df_g.groupby(["leng_bin", "GID_0"])[
                "ct_users"
            ].transform(lambda x: x / x.sum())
            df_g = df_g.sort_values(["dist_bin", "leng_bin"][::-1])

            ## Step 7: Clean lenghts with not enough users
            df_g["gap_clean"] = df_g.apply(
                lambda row: (
                    np.nan
                    if (row["MALE"] < min_u) or (row["FEMALE"] < min_u)
                    else row["gap"]
                ),
                axis=1,
            )

            # df_g['gap_sym_clean'] = df_g.apply(
            #     lambda row: np.nan if (row['MALE'] < min_u) or (row['FEMALE'] < min_u) else row['gap_sym'],
            #     axis = 1)

            # Step 8: Formatting bins as median dist in bin
            df_g["dist_xbins_med"] = df_g["dist_bin"].apply(
                lambda x: round(np.median([x.left, x.right]), 1)
            )

            # Append the result to the list
            bootstrap_samples.append(
                df_g[
                    [
                        "leng_bin",
                        "dist_bin",
                        "GID_0",
                        "dist_xbins_med",
                        "frac_users",
                        "gap_clean",
                    ]
                ].copy()
            )

        # Step BS-2: Create a DataFrame from the bootstrap samples
        df_bs = pd.concat(bootstrap_samples, ignore_index=True)

        # Step 5: Calculate the mean and standard error for each bin
        df_bs_summary = (
            df_bs.groupby(["leng_bin", "dist_bin", "GID_0", "dist_xbins_med"])
            .agg(
                mean_gap=("gap_clean", "mean"),
                mean_frac=("frac_users", "mean"),
                se_gap=(
                    "gap_clean",
                    "std",
                ),  #'ambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
            )
            .reset_index()
        )

        df_c = df_bs_summary[df_bs_summary.columns]
        df_c["leng_desc"] = df_c["leng_bin"].map(dLENG)
        df_c["leng_bin"] = pd.Categorical(
            df_c["leng_bin"], categories=LENG_LABELS, ordered=True
        )
        df_c = df_c.sort_values(by="leng_bin")

        dic_test_ctry[test] = df_c

        # Step 6: Get pooled standard error across countries for each bin
        df_a = (
            df_bs_summary.groupby(["leng_bin", "dist_bin"])
            .agg(
                {
                    "mean_gap": "mean",
                    "se_gap": pooled_se,
                    "mean_frac": ["mean", "std"],
                }
            )
            .reset_index()
        )
        df_a.columns = [
            "leng_bin",
            "dist_bin",
            "mean_gap_mean",
            "se_gap_pooled_se",
            "mean_frac_mean",
            "mean_frac_std",
        ]
        df_a["dist_xbins_med"] = df_a["dist_bin"].apply(
            lambda x: round(np.median([x.left, x.right]), 1)
        )
        df_a["leng_desc"] = df_a["leng_bin"].map(dLENG)
        df_a["leng_bin"] = pd.Categorical(
            df_a["leng_bin"], categories=LENG_LABELS, ordered=True
        )
        df_a = df_a.sort_values(by="leng_bin")

        dic_test[test] = df_a

    # Step 7: Save results by ctry
    save_to = output_path + f"gendergap_tours_length_by_dist_byctry_BS{n_iterations}"
    save_to = save_to + f"_cap{max_val}" if capped_dist else save_to
    save_to = save_to + ".pkl"

    with open(save_to, "wb") as f:
        pickle.dump(dic_test_ctry, f)

    print(f">> saved: {save_to}")

    # Step 7: Save results pooled
    save_to = output_path + f"gendergap_tours_length_by_dist_pooled_BS{n_iterations}"
    save_to = save_to + f"_cap{max_val}" if capped_dist else save_to
    save_to = save_to + ".pkl"

    with open(save_to, "wb") as f:
        pickle.dump(dic_test, f)

    print(f">> saved: {save_to}")


# -- GENDER GAP BY JOURNEY LENGHT, ACTIVITY LEVEL ACROSS TOTAL DURATION BINS ---------
if run_tours_gendergap_leng_duration:

    # Step 0: Define vars
    reward_met = "tour_total_duration"
    nbins = 15
    capped_durat = True
    min_val = 1
    max_val = 24 * 60 * 3  # max days in minutes

    dic_test = {}
    dic_test_ctry = {}
    for test in ["all", "moderate", "inactive", "active"]:
        if test == "all":
            df_jrny = df_tours.dropna(subset=["tour", leng_met])
        elif test == "moderate":
            df_jrny = df_tours[
                df_tours["activity_repertoire_groups"] == "moderate"
            ].dropna(subset=["tour", leng_met])
        elif test == "inactive":
            df_jrny = df_tours[
                df_tours["activity_repertoire_groups"] == "inactive"
            ].dropna(subset=["tour", leng_met])
        elif test == "active":
            df_jrny = df_tours[
                df_tours["activity_repertoire_groups"] == "active"
            ].dropna(subset=["tour", leng_met])

        # print('>> start: run_tours_gendergap_leng_duration for ', reward_met, 'capped:'  , capped_durat)

        # Step BS-1: Define bootstrap parameters
        bootstrap_samples = []

        for _ in range(n_iterations):

            # Step 0: Resample with replacement
            df_jrny_ = df_jrny.sample(frac=1, replace=True)
            df_b = df_jrny_[["gender", "GID_0", leng_met, reward_met, "leng_bin"]]

            # Step 1: convert durat to km
            df_b[reward_met] = df_b[reward_met].apply(lambda x: x / 1000)

            # Step 2: Define duration bins
            durat = df_b[df_b[reward_met] >= min_val][reward_met]

            if capped_durat:
                DIST_BINS = np.logspace(np.log10(min_val), np.log10(max_val), num=nbins)
                df_b["durat_bin"] = pd.cut(
                    df_b[reward_met], bins=DIST_BINS, include_lowest=True
                )

            else:
                MAXv = durat.max()
                DIST_BINS = np.logspace(np.log10(min_val), np.log10(MAXv), num=nbins)
                df_b["durat_bin"] = pd.cut(
                    df_b[reward_met], bins=DIST_BINS, include_lowest=True
                )

            df_b = df_b.sort_values(["durat_bin", "leng_bin"])

            # Step 3: Get count of women/men per defined-bin
            df_g = (
                df_b.groupby(["leng_bin", "durat_bin", "gender", "GID_0"])
                .agg({leng_met: "count"})
                .reset_index()
            )
            df_g = df_g.pivot(
                index=["durat_bin", "leng_bin", "GID_0"],
                columns="gender",
                values=leng_met,
            ).reset_index()

            ## Step 4: Normalize by the total volume of users per gender FOR EACH DURATION BIN.
            levels = ["MALE", "FEMALE"]
            for gen in levels:
                lab = gen[0] + "norm"
                df_g[lab] = df_g.groupby(["durat_bin", "GID_0"])[gen].transform(
                    lambda x: x / x.sum()
                )
                df_g[lab] = df_g[lab].fillna(0)

            # Step 5: Get difference in %M and %F per bin
            df_g["gap"] = df_g["Mnorm"] - df_g["Fnorm"]
            # df_g['gap_sym'] = 2*(df_g['Mnorm'] - df_g['Fnorm']) / (df_g['Mnorm'] + df_g['Fnorm'])

            # Step 6: Get Fraction Users per duration bin for each length bin separately
            df_g["ct_users"] = df_g["MALE"] + df_g["FEMALE"]
            df_g["frac_users"] = df_g.groupby(["leng_bin", "GID_0"])[
                "ct_users"
            ].transform(lambda x: x / x.sum())
            df_g = df_g.sort_values(["durat_bin", "leng_bin"][::-1])

            ## Step 7: Clean lenghts with not enough users
            df_g["gap_clean"] = df_g.apply(
                lambda row: (
                    np.nan
                    if (row["MALE"] < min_u) or (row["FEMALE"] < min_u)
                    else row["gap"]
                ),
                axis=1,
            )

            # df_g['gap_sym_clean'] = df_g.apply(
            #     lambda row: np.nan if (row['MALE'] < min_u) or (row['FEMALE'] < min_u) else row['gap_sym'],
            #     axis = 1)

            # Step 8: Formatting bins as median durat in bin
            df_g["durat_xbins_med"] = df_g["durat_bin"].apply(
                lambda x: round(np.median([x.left, x.right]), 1)
            )

            # Append the result to the list
            bootstrap_samples.append(
                df_g[
                    [
                        "leng_bin",
                        "durat_bin",
                        "GID_0",
                        "durat_xbins_med",
                        "frac_users",
                        "gap_clean",
                    ]
                ].copy()
            )

        # Step BS-2: Create a DataFrame from the bootstrap samples
        df_bs = pd.concat(bootstrap_samples, ignore_index=True)

        # Step 5: Calculate the mean and standard error for each bin
        df_bs_summary = (
            df_bs.groupby(["leng_bin", "durat_bin", "GID_0", "durat_xbins_med"])
            .agg(
                mean_gap=("gap_clean", "mean"),
                mean_frac=("frac_users", "mean"),
                se_gap=(
                    "gap_clean",
                    "std",
                ),  #'ambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
            )
            .reset_index()
        )

        df_c = df_bs_summary[df_bs_summary.columns]
        # df_c['durat_xbins_med'] = df_c['durat_bin'].apply(lambda x : round(np.median([x.left, x.right]),1))
        df_c["leng_desc"] = df_c["leng_bin"].map(dLENG)
        df_c["leng_bin"] = pd.Categorical(
            df_c["leng_bin"], categories=LENG_LABELS, ordered=True
        )
        df_c = df_c.sort_values(by="leng_bin")

        dic_test_ctry[test] = df_c

        # Step 6: Get pooled standard error across countries for each bin
        df_a = (
            df_bs_summary.groupby(["leng_bin", "durat_bin"])
            .agg(
                {
                    "mean_gap": "mean",
                    "se_gap": pooled_se,
                    "mean_frac": ["mean", "std"],
                }
            )
            .reset_index()
        )
        df_a.columns = [
            "leng_bin",
            "durat_bin",
            "mean_gap_mean",
            "se_gap_pooled_se",
            "mean_frac_mean",
            "mean_frac_std",
        ]
        df_a["durat_xbins_med"] = df_a["durat_bin"].apply(
            lambda x: round(np.median([x.left, x.right]), 1)
        )
        df_a["leng_desc"] = df_a["leng_bin"].map(dLENG)
        df_a["leng_bin"] = pd.Categorical(
            df_a["leng_bin"], categories=LENG_LABELS, ordered=True
        )
        df_a = df_a.sort_values(by="leng_bin")

        dic_test[test] = df_a

    # Step 7: Save results
    save_to = output_path + f"gendergap_tours_length_by_durat_byctry_BS{n_iterations}"
    save_to = save_to + f"_cap{max_val}" if capped_durat else save_to
    save_to = save_to + ".pkl"

    with open(save_to, "wb") as f:
        pickle.dump(dic_test_ctry, f)

    # Step 7: Save results
    save_to = output_path + f"gendergap_tours_length_by_durat_pooled_BS{n_iterations}"
    save_to = save_to + f"_cap{max_val}" if capped_durat else save_to
    save_to = save_to + ".pkl"

    with open(save_to, "wb") as f:
        pickle.dump(dic_test, f)

    print(f">> saved: {save_to}")
