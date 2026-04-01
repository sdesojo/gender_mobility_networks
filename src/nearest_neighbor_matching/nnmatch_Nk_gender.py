"""
Define functions for nearest neighbor matching by N and k to assess gender differences in user-month-level network metrics.

This script allows analyses across: all users, activity groups, and countries.

"""

import pandas as pd
import pickle as pickle
import numpy as np
import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def knn_matching_n_nu_mf2fm(
    df,
    RUN_VARS=[],
    shuffled_gender=False,
    min_nnu=1,
    max_dist=False,
):
    """
    Perform nearest neighbor matching between male and female users based on visit and location counts.

    This function supports:
    - Optional shuffling of gender labels for permutation testing.
    - Bidirectional matching: males to nearest females (M2F) and unmatched females to nearest males (F2M).
    - Matching is performed using standardized ["n_visits", "n_locations"] features.
    - Returns a DataFrame of matched pairs with calculated differences in specified network metrics.
    """

    #### 0. PREPARE DATAFRAMES: Split MALES/FEMALES
    # Filter to case study
    df_ = df.copy()

    # Transform N, Nu to integers, and keep main_var > 0
    df_["n_visits"] = df_["visits"].astype(int)
    df_["n_locations"] = df_["locations"].astype(int)

    for main_var in RUN_VARS + ["n_visits", "n_locations"]:
        df_ = df_[(df_[main_var] > 0)]

    for ctrl_var in ["n_visits", "n_locations"]:
        df_ = df_[(df_[ctrl_var] > min_nnu)]
        # for at least 3 locations visited?

    # Permutation gender labels
    if shuffled_gender:
        shuffled_lab = np.random.permutation(df_["gender"])
        df_["gender"] = shuffled_lab

    ## Split by Gender
    df_male = (
        df_[df_["gender"] == "MALE"].drop(columns=["gender"]).reset_index(drop=True)
    )
    df_female = (
        df_[df_["gender"] == "FEMALE"].drop(columns=["gender"]).reset_index(drop=True)
    )

    ## Update all features
    suffix = "_male"
    df_male.columns = [
        col + suffix if col != "useruuid" else "user" + suffix
        for col in df_male.columns
    ]
    suffix = "_female"
    df_female.columns = [
        col + suffix if col != "useruuid" else "user" + suffix
        for col in df_female.columns
    ]

    #### 1. PAIR ALL MALES WITH CLOSEST FEMALE
    # Convert dataframes to numpy arrays and scale
    X_male = df_male[["n_visits_male", "n_locations_male"]].values
    X_female = df_female[["n_visits_female", "n_locations_female"]].values

    scaler = StandardScaler()
    X_male_normalized = scaler.fit_transform(X_male)
    X_female_normalized = scaler.transform(X_female)
    # print('>> gender normalized')

    # Initialize nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean")
    # Fit model on female data
    nn_model.fit(X_female_normalized)
    # Find nearest neighbor for each male
    distances, indices = nn_model.kneighbors(X_male_normalized)
    # print('>> nn calculated')

    # Create a dataframe to store matched pairs
    matched_df_male = pd.DataFrame(
        {
            "user_male": df_male["user_male"],
            "start_month_male": df_male["start_month_male"],
        }
    ).reset_index()

    matched_df_female = pd.DataFrame(
        {
            "user_female": df_female.iloc[indices.flatten()]["user_female"],
            "start_month_female": df_female.iloc[indices.flatten()][
                "start_month_female"
            ],
            "distance": distances.flatten(),
        }
    ).reset_index()

    m2f_matched_df = pd.merge(
        matched_df_male, matched_df_female, left_index=True, right_index=True
    ).drop(columns=["index_x", "index_y"])
    # print(matched_df.shape[0])

    # Add distance threshold
    # if max_dist != False:
    #     m2f_matched_df = m2f_matched_df[m2f_matched_df["distance"] < max_dist]
    # print(f'>> matched and distance filtered <{max_dist}: ,', matched_df.shape[0])

    ##### 2. FIND ALL FEMALES NOT YET MATCHED TO ANY MALE
    # Find missing females
    females_already_included = m2f_matched_df[
        ["user_female", "start_month_female"]
    ].drop_duplicates()
    # Perform a left join and filter out matches
    df_F_join = df_female.merge(
        females_already_included,
        on=["user_female", "start_month_female"],
        how="left",
        indicator=True,
    )
    df_female_missing = (
        df_F_join[df_F_join["_merge"] == "left_only"]
        .drop(columns=["_merge"])
        .reset_index()
    )

    ##### 3. PAIR ALL MISSING FEMALES WITH CLOSEST MALE

    # Convert dataframes to numpy arrays and scale
    X_male = df_male[["n_visits_male", "n_locations_male"]].values
    X_female = df_female_missing[["n_visits_female", "n_locations_female"]].values

    scaler = StandardScaler()
    X_female_normalized = scaler.fit_transform(X_female)
    X_male_normalized = scaler.transform(X_male)
    # print('>> gender normalized')

    # Initialize nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean")
    # Fit model on male data
    nn_model.fit(X_male_normalized)
    # Find nearest neighbor for each female
    distances, indices = nn_model.kneighbors(X_female_normalized)
    # print('>> nn calculated')

    # Create a dataframe to store matched pairs
    matched_df_female = pd.DataFrame(
        {
            "user_female": df_female_missing["user_female"],
            "start_month_female": df_female_missing["start_month_female"],
        }
    ).reset_index()

    matched_df_male = pd.DataFrame(
        {
            "user_male": df_male.iloc[indices.flatten()]["user_male"],
            "start_month_male": df_male.iloc[indices.flatten()]["start_month_male"],
            "distance": distances.flatten(),
        }
    ).reset_index()

    # f2m_matched_df = pd.merge(matched_df_male, matched_df_female, left_index=True, right_index=True).drop(columns = ['index_x', 'index_y'])
    f2m_matched_df = pd.merge(
        matched_df_female, matched_df_male, left_index=True, right_index=True
    ).drop(columns=["index_x", "index_y"])

    # # Add distance threshold
    # if max_dist != False:
    #     f2m_matched_df = f2m_matched_df[f2m_matched_df["distance"] < max_dist]

    # print(f'>> matched and distance filtered <{max_dist}: ,', matched_df.shape[0])

    #### 3. MERGE AND GET DELTA BETWEEN PAIRS
    COLS = [
        "user_female",
        "start_month_female",
        "user_male",
        "start_month_male",
    ]  # dropping distance and sorting
    m2f_matched_df = m2f_matched_df[COLS]
    fm2_matched_df = f2m_matched_df[COLS]
    matched_df = pd.concat([m2f_matched_df, f2m_matched_df])

    # print(">> Duplicates:", matched_df[matched_df.duplicated()].shape[0])
    matched_df = matched_df.drop_duplicates()

    # Merge matched pairs with main dataframe to get main features
    matched_df = pd.merge(matched_df, df_male, on=["user_male", "start_month_male"])
    # print('>> males added', '\n', matched_df.columns)

    matched_df = pd.merge(
        matched_df, df_female, on=["user_female", "start_month_female"]
    )
    # print('>> females added', '\n', matched_df.columns)

    ## Get delta main var
    for main_var in RUN_VARS + ["locations", "visits"]:
        matched_df["abs_dif_" + main_var] = (
            matched_df[main_var + "_male"] - matched_df[main_var + "_female"]
        )

        matched_df["rel_dif_" + main_var] = (
            matched_df[main_var + "_male"] - matched_df[main_var + "_female"]
        ) / (matched_df[main_var + "_male"] + matched_df[main_var + "_female"])

        # matched_df["relF_dif_" + main_var] = (
        #     matched_df[main_var + "_male"] - matched_df[main_var + "_female"]
        # ) / matched_df[main_var + "_female"]

        # matched_df["abs_rel_dif_" + main_var] = np.abs(
        #     matched_df[main_var + "_male"] - matched_df[main_var + "_female"]
        # ) / (matched_df[main_var + "_male"] + matched_df[main_var + "_female"])

        matched_df["abs_abs_dif_" + main_var] = np.abs(
            matched_df[main_var + "_male"] - matched_df[main_var + "_female"]
        )

    # return m2f_matched_df, f2m_matched_df, matched_df
    return matched_df


# def get_aggregated_stats(df_i, RUN_VARS=[], GAP_METRICS=["abs_dif_", "rel_dif_"]):
#     D_METRICS = [d + m for m, d in itertools.product(RUN_VARS, GAP_METRICS)]
#     dic = {}
#     for dmlab in D_METRICS:
#         dic[dmlab] = {
#             "mean": np.mean(df_i[dmlab]),
#             "mean_pos0": np.mean(df_i[df_i[dmlab] >= 0][dmlab]),
#             "mean_neg0": np.mean(df_i[df_i[dmlab] <= 0][dmlab]),
#             "ct_pos": df_i[df_i[dmlab] > 0].shape[0],
#             "ct_neg": df_i[df_i[dmlab] < 0].shape[0],
#             "ct_0": df_i[df_i[dmlab] == 0].shape[0]

#         }

#     return dic


# # --------------------------- core runners ---------------------------
# def _it_nnmatching(
#     df, RUN_VARS=[], GAP_METRICS=["abs_dif_", "rel_dif_"],
#     get_resample=False, min_nnu=3, max_delta_nu=None,
#     # max_delta_nu=None, test=None
# ):

#     ## DO WE WANT TO RESAMPLE? options: by population, with equal weight, or resample the original sample as is.
# #    # get resmaple based on population weights
# #    if get_resample:
#         # dic_frac_pop = {ctry: utils.get_fract_pop(ctry) for ctry in CTRY}
# #        df_i = utils.get_resample_by_pop_weights(df)
# #     else:
# #        df_i = df.copy()


#     # get true pairs
#     df_match_true = knn_matching_n_nu_mf2fm(
#         df_i, RUN_VARS, shuffled_gender=False, min_nnu=min_nnu,
#     )

#     # get shuffled gender pairs
#     df_match_shuffled = knn_matching_n_nu_mf2fm(
#         df_i, RUN_VARS, shuffled_gender=True, min_nnu=min_nnu,
#     )

#     # drop pairs with too large k delta (ensures matching to similar users, ==1 is equivalent to exact matching)
#     if max_delta_nu is not None:
#         df_match_true = df_match_true[df_match_true["abs_abs_dif_" + "locations"] <= max_delta_nu]
#         df_match_shuffled = df_match_shuffled[df_match_shuffled["abs_abs_dif_" + "locations"] <= max_delta_nu]

#     # get aggregated stats
#     res_i_true = get_aggregated_stats(df_match_true, RUN_VARS, GAP_METRICS)
#     res_i_shuffled = get_aggregated_stats(df_match_shuffled, RUN_VARS, GAP_METRICS)

#     return {
#         "True": res_i_true,
#         "Shuffled": res_i_shuffled,
#     }

# def run_nnmatching_byNk(
#     df, RUN_VARS=[], GAP_METRICS=["abs_dif_", "rel_dif_"],
#     get_resample=False, min_nnu=3, max_delta_nu=None,
#     n_samples=1000,
# ):
#     """
#     Run nearest neighbor matching with/without resampling by population weights.
#     """
#     D_METRICS = [d + m for m, d in itertools.product(RUN_VARS, GAP_METRICS)]
#     AGG_METRICS = ["mean", "mean_pos0", "mean_neg0", "ct_pos", "ct_neg", "ct_0"]

#     dic_res = {
#         "True": {dmlab: {aggmet: [] for aggmet in AGG_METRICS} for dmlab in D_METRICS},
#         "Shuffled": {dmlab: {aggmet: [] for aggmet in AGG_METRICS} for dmlab in D_METRICS},
#     }

#     # Run n_samples iterations
#     for _ in range(n_samples):

#         res_it = _it_nnmatching(
#             df, RUN_VARS, GAP_METRICS,
#             get_resample=get_resample, min_nnu=min_nnu, max_delta_nu=max_delta_nu,
#         )

#         for key in res_it.keys():
#             for dmlab in D_METRICS:
#                 for calc in AGG_METRICS:
#                     dic_res[key][dmlab][calc].append(res_it[key][dmlab][calc])

#     return dic_res
