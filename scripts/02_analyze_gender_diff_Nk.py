"""
    Script to analyze gender differences in mobility metrics (Nk) across countries and samples.
    This anlaysis will be visualized in figure 1 of the paper.
    
    Analysis includes:
    - Histograms of Nk distributions
    - Bootstrap medians and standard errors
    - Gender differences in k by activity deciles
    - Share of men/women by N,k deciles
    - KS statistic
    - Bootrstap relative dispersion (RCV)

"""

import os
import pandas as pd
import pickle
import numpy as np

from utils.utils import GENDER, CTRY, perc2quant

import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# DEFINE INPUT DATA, OUTPUT PATH, AND ANALYSIS TO RUN
# ---------------------------------------------------------------------

# --- Input / Output Paths ---
input_path = ''
output_path = ''

# --- Load ---
fname_metrics = 'user_metrics_all.csv'
df = pd.read_csv(input_path + fname_metrics)

get_rsC = True
fname_metrics = 'user_metrics_all_rsC.csv'
df_rs_c = pd.read_csv(input_path + fname_metrics)


# --- Define variables to run ---
MLAB = ["visits", "locations"]

# --- Define analysis to run ---
run_histograms = False
run_bootstrap = False
run_deciles_Nk = False
run_deciles_gap = False
run_ks = False
run_rcv = True
run_frac_users_bin = True

print(f"> Input data: {input_path + fname_metrics}")
print(f"> Output path: {output_path}")

print(f"> Get resampled results: {get_rsC}")
print(f"> Run histograms: {run_histograms}")
print(f"> Run gap k by N deciles: {run_deciles_Nk}")
print(f"> Run gap in deciles: {run_deciles_gap}")
print(f"> Run bootstrap: {run_bootstrap}")
print(f"> Run KS test: {run_ks}")
print(f"> Run RCV: {run_rcv}")


# print(f"> Metrics to analyze: {MLAB}")
# print(f"> Anlaysis for {GENDER} and {CTRY}")
print("---------------------------------------------------------------------")

# ---------------------------------------------------------------------
# START ANALYSIS
# ---------------------------------------------------------------------

## --- Historgrams of Nk distributions ---
if run_histograms:

    from statistical_analysis.basic_stats import get_hist_xxyy

    # Define bins and save
    print(">> Start histograms")
    dimension = "gender"
    hist_dict = {
        mlab: {ctry: {} for ctry in ["all_ctry", "all_ctry_wequalC"] + CTRY}
        for mlab in MLAB
    }
    nbins = 10  # 12

    for mlab in MLAB:
        for stat in ["MALE", "FEMALE"]:
            v = df[(df[dimension] == stat) & (df[mlab] > 0)][mlab].dropna().values
            xx, yy = get_hist_xxyy(v, nbins)
            hist_dict[mlab]["all_ctry"][stat] = (xx, yy)

            if get_rsC:
                vrs_c = (
                    df_rs_c[(df_rs_c[dimension] == stat) & (df_rs_c[mlab] > 0)][mlab]
                    .dropna()
                    .values
                )
                xx, yy = get_hist_xxyy(vrs_c, nbins)
                hist_dict[mlab]["all_ctry_wequalC"][stat] = (xx, yy)

            for ctry in CTRY:
                df_c = df[(df["GID_0"] == ctry)]
                v_c = (
                    df_c[(df[dimension] == stat) & (df_c[mlab] > 0)][mlab]
                    .dropna()
                    .values
                )
                xx, yy = get_hist_xxyy(v_c, nbins)
                hist_dict[mlab][ctry][stat] = (xx, yy)

    # print(hist_dict.keys())
    # print(hist_dict[mlab].keys())

    # Save
    fname = f"fig1_hist_dict_nbins{nbins}.pkl"
    with open(output_path + fname, "wb") as f:
        pickle.dump(hist_dict, f)
    print(f">> Saved histograms in {output_path + fname}")

## --- Bootstrap medians ---
if run_bootstrap:

    from statistical_analysis.basic_stats import get_bs_median_se, agg_mean_se_results

    print(">> Start bootstrap")
    n_samples = 1000  # 10000
    dic_gen = {gen: {mlab: {} for mlab in MLAB} for gen in GENDER}
    for gen in GENDER:
        for mlab in MLAB:
            BS_MEAN_MD = []
            BS_SE_MED = []
            for ctry in CTRY:
                data = df[(df["GID_0"] == ctry) & (df["gender"] == gen)]

                MED, SE = get_bs_median_se(data, mlab, n_samples)
                dic_gen[gen][mlab][ctry] = {"med": MED, "se": SE}
                BS_MEAN_MD.append(MED)
                BS_SE_MED.append(SE)
                # print(gen, mlab, ctry)

            MED_AGG, SE_AGG = agg_mean_se_results(
                BS_MEAN_MD, BS_SE_MED, pop_weight=False, CTRY=CTRY
            )
            dic_gen[gen][mlab]["agg_equalweight"] = {"med": MED_AGG, "se": SE_AGG}

            MED_AGG, SE_AGG = agg_mean_se_results(
                BS_MEAN_MD, BS_SE_MED, pop_weight=True, CTRY=CTRY
            )
            dic_gen[gen][mlab]["agg_popweight"] = {"med": MED_AGG, "se": SE_AGG}

    # Save
    fname = "fig1_bs_med.pkl"
    with open(output_path + fname, "wb") as f:
        pickle.dump(dic_gen, f)
    print(f">> Saved bootstrap results in {output_path + fname}")


## --- Gender diff in Repertoire by activity deciles ---
if run_deciles_Nk:

    from statistical_analysis.basic_stats import (
        allocate_mlab_percentiles,
        get_bs_median_se,
        apply_bootstrap,
    )

    print(">> Start deciles gap by N deciles")
    nquant = 10
    quant_var = "visits"
    var = "locations"

    dic_Nkquant = {ctry: {} for ctry in ["all_ctry", "all_ctry_wequalC"] + CTRY}

    ## Run across countries
    df_ = allocate_mlab_percentiles(df, nperc=nquant, MLAB=[quant_var])

    df_Aq = (
        df_.groupby(["q10_" + quant_var, "gender"])
        .apply(apply_bootstrap, var)
        .reset_index()
    )
    df_Aq_pv = df_Aq.pivot(
        index="q10_" + quant_var, values=["median", "standard_error"], columns="gender"
    ).reset_index()
    df_Aq_pv["diff_med"] = df_Aq_pv["median"]["MALE"] - df_Aq_pv["median"]["FEMALE"]
    df_Aq_pv["diff_se"] = np.sqrt(
        df_Aq_pv["standard_error"]["FEMALE"] ** 2
        + df_Aq_pv["standard_error"]["MALE"] ** 2
    )  # Propagated SE
    df_Aq_pv["q10_" + quant_var + "_lab"] = df_Aq_pv["q10_" + quant_var].apply(
        lambda x: perc2quant[x]
    )
    dic_Nkquant["all_ctry"] = df_Aq_pv

    ## Run across countries - resampled
    if get_rsC:
        df_ = allocate_mlab_percentiles(df_rs_c, nperc=10, MLAB=[quant_var])

        df_Aq = (
            df_.groupby(["q10_" + quant_var, "gender"])
            .apply(apply_bootstrap, var)
            .reset_index()
        )
        df_Aq_pv = df_Aq.pivot(
            index="q10_" + quant_var,
            values=["median", "standard_error"],
            columns="gender",
        ).reset_index()
        df_Aq_pv["diff_med"] = df_Aq_pv["median"]["MALE"] - df_Aq_pv["median"]["FEMALE"]
        df_Aq_pv["diff_se"] = np.sqrt(
            df_Aq_pv["standard_error"]["FEMALE"] ** 2
            + df_Aq_pv["standard_error"]["MALE"] ** 2
        )  # Propagated SE
        df_Aq_pv["q10_" + quant_var + "_lab"] = df_Aq_pv["q10_" + quant_var].apply(
            lambda x: perc2quant[x]
        )
        dic_Nkquant["all_ctry_wequalC"] = df_Aq_pv

    ## Run by country
    for ctry in CTRY:
        df_c = df[df["GID_0"] == ctry]
        df_ = allocate_mlab_percentiles(df_c, nperc=10, MLAB=[quant_var])

        df_Aq = (
            df_.groupby(["q10_" + quant_var, "gender"])
            .apply(apply_bootstrap, var)
            .reset_index()
        )
        df_Aq_pv = df_Aq.pivot(
            index="q10_" + quant_var,
            values=["median", "standard_error"],
            columns="gender",
        ).reset_index()
        df_Aq_pv["diff_med"] = df_Aq_pv["median"]["MALE"] - df_Aq_pv["median"]["FEMALE"]
        df_Aq_pv["diff_se"] = np.sqrt(
            df_Aq_pv["standard_error"]["FEMALE"] ** 2
            + df_Aq_pv["standard_error"]["MALE"] ** 2
        )  # Propagated SE
        df_Aq_pv["q10_" + quant_var + "_lab"] = df_Aq_pv["q10_" + quant_var].apply(
            lambda x: perc2quant[x]
        )

        dic_Nkquant[ctry] = df_Aq_pv

    # Save and import directly from the notebook
    FIG_PATH = "/work/user/sdsc/gender_dif_entropy/data_for_figures/"
    with open(FIG_PATH + "fig1_gend_dif_k_Nquant.pkl", "wb") as f:
        pickle.dump(dic_Nkquant, f)

    # Save
    fname = "fig1_diff_k_by_Ndecile.pkl"
    with open(output_path + fname, "wb") as f:
        pickle.dump(dic_Nkquant, f)
    print(f">> Saved Nk decile results in {output_path + fname}")


## --- Share of men/women by deciles ---
if run_deciles_gap:

    from statistical_analysis.basic_stats import get_deltau_deciles

    print(f">> Start fracu by decile")

    dic_fracu = {
        mlab: {ctry: {} for ctry in ["all_ctry", "all_ctry_wequalC"] + CTRY}
        for mlab in MLAB
    }
    for mlab in MLAB:
        for ctry in ["all_ctry", "all_ctry_wequalC"] + CTRY:
            if ctry == "all_ctry":
                df_ = df.copy()
            elif ctry == "all_ctry_wequalC":
                if get_rsC:
                    df_ = df_rs_c.copy()
                else:
                    continue  # Skip if no resampled data available
            else:
                df_ = df[df["GID_0"] == ctry].copy()

            df_ = df_[(df_[mlab] > 0)]
            dic_, quantile_values = get_deltau_deciles(
                df_, mlab, nquant=10, dimension="gender"
            )
            dic_fracu[mlab][ctry]["res"] = dic_
            dic_fracu[mlab][ctry]["quantiles"] = quantile_values

    # Save
    fname = "fig1_fracu_by_decile.pkl"
    with open(output_path + fname, "wb") as f:
        pickle.dump(dic_fracu, f)
    print(f">> Saved fracu by decile results in {output_path + fname}")


## --- KS statistic ---
if run_ks:
    from statistical_analysis.basic_stats import get_KS_true_shuf

    print(f">> Start KS test")
    n_samples = 1000
    dic_ks = {ctry: {} for ctry in ["all_ctry", "all_ctry_wequalC"] + CTRY}
    for ctry in ["all_ctry", "all_ctry_wequalC"] + CTRY:
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

        ks_true, ks_rsh = get_KS_true_shuf(
            df_,
            sel_MLAB=MLAB,
            var="gender",
            dimension=GENDER,
            n=n_samples,
        )

        dic_ks[ctry]["KS_true"] = ks_true
        dic_ks[ctry]["KS_shuff"] = ks_rsh

    # Save
    fname = "fig1_ks_test.pkl"
    with open(output_path + fname, "wb") as f:
        pickle.dump(dic_ks, f)
    print(f">> Saved KS test results in {output_path + fname}")


## --- RCV ---
if run_rcv:
    from statistical_analysis.basic_stats import (
        compute_rcvq,
        compute_rcvm,
        bs_by_metric,
        agg_mean_se_results,
    )

    print(f">> Start RCV test")

    # Define the metrics to compute
    METRICS = {"RCVQ": compute_rcvq, "RCVM": compute_rcvm}

    n_samples = 1000
    dic_gen = {
        gen: {metric: {mlab: {} for mlab in MLAB} for metric in METRICS}
        for gen in GENDER
    }

    for gen in GENDER:
        for metric_name, metric_func in METRICS.items():
            for mlab in MLAB:
                BS_MEAN_MD = []
                BS_SE_MED = []
                for ctry in CTRY:
                    data = df[(df.GID_0 == ctry) & (df.gender == gen)]

                    mean, se, _ = bs_by_metric(
                        data, mlab, metric_func, n_samples=n_samples
                    )
                    dic_gen[gen][metric_name][mlab][ctry] = {"mean": mean, "se": se}

                    BS_MEAN_MD.append(mean)
                    BS_SE_MED.append(se)
                    # print(f"{gen} | {metric_name} | {mlab} | {ctry}")

                # Aggregate across countries (equal weight)
                med_agg, se_agg = agg_mean_se_results(
                    BS_MEAN_MD, BS_SE_MED, pop_weight=False, CTRY=CTRY
                )
                dic_gen[gen][metric_name][mlab]["agg_equalweight"] = {
                    "mean": med_agg,
                    "se": se_agg,
                }

                # Aggregate across countries (population weight)
                med_agg, se_agg = agg_mean_se_results(
                    BS_MEAN_MD, BS_SE_MED, pop_weight=True, CTRY=CTRY
                )
                dic_gen[gen][metric_name][mlab]["agg_popweight"] = {
                    "mean": med_agg,
                    "se": se_agg,
                }

    # Save
    fname = "fig1_bs_rcv.pkl"
    with open(output_path + fname, "wb") as f:
        pickle.dump(dic_gen, f)
    print(f">> Saved RCV results in {output_path + fname}")


## --- Fraction users per bin

if run_frac_users_bin:
    # ── 1. Build bin counts ─────────────────────────────────────────────────────
    df_ = df.copy()
    total_users = df_["useruuid"].nunique()

    bin_counts = (
        df_.groupby(["activity_deciles", "repertoire_deciles"])["useruuid"]
        .nunique()
        .reset_index(name="n")
    )
    bin_counts["pct"] = bin_counts["n"] / total_users * 100

    grid_pct = np.zeros((10, 10))
    for _, row in bin_counts.iterrows():
        r = int(row["repertoire_deciles"]) - 1
        a = int(row["activity_deciles"]) - 1
        grid_pct[r, a] = row["pct"]

    # Save
    fname = "fig1_si_fract_users_ra.pkl"
    with open(output_path + fname, "wb") as f:
        pickle.dump(grid_pct, f)
    print(f">> Saved fraction users results in {output_path + fname}")

    # ── 2. Gender delta grid ────────────────────────────────────────────────────
    # Requires a 'gender' column ('M' / 'F')
    male = df[df["gender"] == "MALE"]
    female = df[df["gender"] == "FEMALE"]

    def gender_bin_pct(sub, total):
        counts = (
            sub.groupby(["activity_deciles", "repertoire_deciles"])["useruuid"]
            .nunique()
            .reset_index(name="n")
        )
        counts["pct"] = counts["n"] / total * 100
        g = np.zeros((10, 10))
        for _, row in counts.iterrows():
            g[int(row["repertoire_deciles"]) - 1, int(row["activity_deciles"]) - 1] = (
                row["pct"]
            )
        return g

    grid_m = gender_bin_pct(male, male["useruuid"].nunique())
    grid_f = gender_bin_pct(female, female["useruuid"].nunique())
    grid_delta = grid_m - grid_f

    # Save
    fname = "fig1_si_fract_users_gender_delta_ra.pkl"
    with open(output_path + fname, "wb") as f:
        pickle.dump(grid_delta, f)
    print(f">> Saved gender delta results in {output_path + fname}")
