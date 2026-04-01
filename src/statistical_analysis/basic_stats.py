import numpy as np
import pandas as pd
from scipy import stats
import tqdm

from utils.utils import get_fract_pop, perc2quant, GENDER


# -- HISTOGRAM ----------------------------------------
# -- Get Histograms BINS and FREQUENCY
def get_hist_xxyy(values, nbins=15, linspace=True):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if linspace:
        bins = np.linspace(0, max(values), nbins)
    else:
        bins = np.logspace(0, np.log10(max(values)), 10)

    hist, edges = np.histogram(values, bins=bins, density=True)
    x = (edges[1:] + edges[:-1]) / 2
    xx, yy = zip(*[(i, j) for (i, j) in zip(x, hist) if j > 0])
    return xx, yy


# -- QUANILES USER SPLIT By GROUP ----------------------------------------
# -- Get quantiles differences in fraction of users by deciles
def allocate_mlab_percentiles(df_, nperc=6, MLAB=[]):
    for mlab in MLAB:
        aaa, arr = pd.qcut(
            df_[df_[mlab] > 0][mlab].dropna(),
            nperc,
            retbins=True,
            labels=False,
            duplicates="raise",
        )
        df_["perc_" + mlab] = 100 * (aaa.astype(int) + 1) / nperc
        df_[f"q{str(nperc)}_" + mlab] = aaa.astype(int) + 1
        # df_['percbins_'+mlab] = arr#.astype('int')
    return df_.dropna()


# -- Get gender differences in fraction of users by deciles
def get_deltau_deciles(
    df_, mlab, nquant=10, dimension="gender", GENDER=GENDER, perc2quant=perc2quant
):
    """
    Get the fraction of users in each percentile
    """
    ## Get quantiles
    df_[f"quantiles"] = (pd.qcut(df_[mlab], nquant, labels=False) + 1).apply(
        lambda x: perc2quant[x]
    )
    quantile_values = df_[mlab].quantile([i / nquant for i in range(nquant + 1)]).values

    # Count #users per group in each percentile
    df_g = df_.groupby([f"quantiles", dimension]).agg({mlab: "count"}).reset_index()
    df_g = df_g.pivot(index=f"quantiles", columns=dimension, values=mlab).reset_index()

    # Normalize by the total volume of users per gender
    for i_gen in range(2):
        lab = GENDER[i_gen][0] + "norm"
        df_g[lab] = df_g[GENDER[i_gen]].transform(lambda x: x / x.sum())

    df_g["gap"] = df_g["Mnorm"] - df_g["Fnorm"]

    return df_g.set_index("quantiles").to_dict(), quantile_values


## -- DISPERSION METRICS ----------------------------------------
# (see https://pmc.ncbi.nlm.nih.gov/articles/PMC9196089/#S006)
def compute_rcvq(data):
    """Computes the robust coefficient of variation using the IQR (RCVQ)."""
    q1 = np.percentile(data, 25)
    median = np.median(data)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return 0.75 * iqr / median if median != 0 else np.nan


def compute_rcvm(data):
    """Computes the robust coefficient of variation using the MAD (RCVM)."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return 1.4826 * mad / median if median != 0 else np.nan


# -- BOOTSTRAP MEDIANS ----------------------------------------
## Get bs mean and se of the median
def get_bs_median_se(df, mlab, n_samples):
    values = df[mlab].values
    # Generate bootstrap samples and compute the statistic
    bs_medians = []
    for i in range(n_samples):
        sample = np.random.choice(values, size=len(values), replace=True)
        bs_medians.append(np.median(sample))

    bs_mean_med = np.mean(bs_medians)
    bs_se_med = np.std(bs_medians)
    return bs_mean_med, bs_se_med


# Apply for dataframes
def apply_bootstrap(group, var, n_samples=1000):
    bs_mean_med, bs_se_med = get_bs_median_se(group, var, n_samples)
    return pd.Series({"median": bs_mean_med, "standard_error": bs_se_med})


## Aggregate the mean and se of the median across countries, with or without population weights
def agg_mean_se_results(BS_MEAN_MD, BS_SE_MED, pop_weight=True, CTRY=[]):
    if pop_weight:
        dicMDCT = {c: md for c, md in zip(CTRY, BS_MEAN_MD)}
        dicSDCT = {c: md for c, md in zip(CTRY, BS_SE_MED)}

        mean_bs_mean_md = sum(
            [get_fract_pop(ctry_id) * dicMDCT[ctry_id] for ctry_id in CTRY]
        )
        se_bs_se_md = np.sqrt(
            sum([get_fract_pop(ctry_id) * (dicSDCT[ctry_id] ** 2) for ctry_id in CTRY])
        )
        return mean_bs_mean_md, se_bs_se_md

    else:
        mean_bs_mean_md = sum(BS_MEAN_MD) / len(BS_MEAN_MD)
        se_bs_se_md = np.sqrt(sum([se**2 for se in BS_SE_MED]) / len(BS_SE_MED))
        return mean_bs_mean_md, se_bs_se_md


## Get Bootstrap to given metruic function (e.g. RCVQ, RCVM)
def bs_by_metric(df, column, metric_func, n_samples=1000, random_state=None):
    """Bootstrap CI for a single metric function (e.g. RCVQ, RCVM)."""
    rng = np.random.default_rng(random_state)
    values = df[column].dropna().values
    bs_results = []

    for _ in range(n_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        stat = metric_func(sample)
        bs_results.append(stat)

    bs_mean = np.mean(bs_results)
    bs_se = np.std(bs_results, ddof=1)
    return bs_mean, bs_se, bs_results


## Get GENDRR GAP Bootstrap to given metruic function (e.g. RCVQ, RCVM)
def bootstrap_metric_gap(
    df1, df2, column, metric_func, n_samples=1000, random_state=None
):
    """Bootstrap relative and symmetric gap between two samples for a metric."""
    rng = np.random.default_rng(random_state)
    values1 = df1[column].dropna().values
    values2 = df2[column].dropna().values

    bs_gap_symmetric = []
    bs_gap_relative = []

    for _ in range(n_samples):
        sample1 = rng.choice(values1, size=len(values1), replace=True)
        sample2 = rng.choice(values2, size=len(values2), replace=True)

        m1 = metric_func(sample1)
        m2 = metric_func(sample2)

        if m1 + m2 != 0:
            bs_gap_symmetric.append((m1 - m2) / ((m1 + m2) / 2))

    return {
        "gap_mean": np.mean(bs_gap_symmetric),
        "gap_se": np.std(bs_gap_symmetric, ddof=1),
        # "all_gaps": bs_gap_symmetric,
    }


# -- KOLMOGROV-SMIRNOV ----------------------------------------


def KS_less_greater(df, sel_MLAB=[], var="gender", dimension=["MALE", "FEMALE"]):
    RESd = {}
    for m in sel_MLAB:
        e1 = df[(df[var] == dimension[0]) & (df[m] > 0)][m].dropna()
        e2 = df[(df[var] == dimension[1]) & (df[m] > 0)][m].dropna()

        res_less = stats.ks_2samp(e1, e2, alternative="less")
        res_greater = stats.ks_2samp(e1, e2, alternative="greater")

        RESd[m] = {
            "less": {
                "stat": res_less.statistic,
                "pvalue": res_less.pvalue,
                "sign": str(res_less.statistic_sign),
                "loc": str(res_less.statistic_location),
            },
            "greater": {
                "stat": res_greater.statistic,
                "pvalue": res_greater.pvalue,
                "sign": str(res_greater.statistic_sign),
                "loc": str(res_greater.statistic_location),
            },
        }
    return RESd


## SHUFFLE KS
def KS_shuffled(df, sel_MLAB=[], var="gender", dimension=["MALE", "FEMALE"]):
    df_ = df.copy()
    shuffled_lab = np.random.permutation(df_[var])
    df_[var] = shuffled_lab

    # try: RESd_sh = KS_less_greater(df_, sel_MLAB=sel_MLAB, var=var, dimension = dimension)
    # except: print('error in KS function')
    RESd_sh = KS_less_greater(df_, sel_MLAB=sel_MLAB, var=var, dimension=dimension)
    return RESd_sh


## RETURN KS True, Shuffled
def get_KS_true_shuf(df, sel_MLAB=[], var="gender", dimension=["MALE", "FEMALE"], n=10):
    ks_true = KS_less_greater(df, sel_MLAB=sel_MLAB, var=var, dimension=dimension)
    ks_rsh = [
        KS_shuffled(df, sel_MLAB=sel_MLAB, var=var, dimension=dimension)
        for i in tqdm.tqdm(range(n))
    ]
    return ks_true, ks_rsh
