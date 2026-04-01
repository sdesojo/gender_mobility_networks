"""
Identify journeys/tours from monthly sequences of stop-level mobility data.

This script:
1. Loads stop data partitioned by user groups.
2. Converts timestamps to local time and applies temporal aggregation.

Execution is controlled via a YAML configuration file and can be run
for all user groups or a single specified group.

Typical usage:
    python ./scripts/05_compute_sequences_tours.py --config ./configs/monthly_met_all_config.yaml
    python ./scripts/05_compute_sequences_tours.py --config ./configs/monthly_met_all_config.yaml --user_group user_group=00
"""

import argparse
import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from tqdm import tqdm
import time
import matplotlib as mpl
import pandas as pd

from howde import HoWDe_labelling

from utils.utils import start_spark
from compute_seq_tours.io import (
    list_user_groups,
    load_config,
    local_time_resolution,
)
from compute_seq_tours.sequences_tours import (
    get_stop_sequences,
    udf_clean_seq_consec_stops,
    udf_extract_tours_and_keystones,
    udf_get_main_keystone,
    udf_get_seq_distances,
    add_jrny_reward_columns,
    udf_summarize_tours,
    get_length_bin,
)


import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


def main():
    time_start = time.time()

    parser = argparse.ArgumentParser(description="Compute mobility metrics from stops.")
    parser.add_argument("--config", required=True, help="Path to task config YAML")
    parser.add_argument(
        "--user_group",
        required=False,
        default=None,
        help="Optional: process only one user_group, e.g. user_group=05",
    )
    args = parser.parse_args()

    # ---- Load task config ----
    cfg = load_config(args.config)
    app_name = cfg.get("app_name", "mobmet")
    job_cfg = cfg["job_spark"]
    agg_cfg = cfg["aggregation"]
    seqtours_cfg = cfg["sequnces_tours"]
    howde_cfg = cfg["home_work_detection"]

    input_path = cfg["paths"]["compute_seq_tours"]["input_path"]
    demo_path = cfg["paths"]["compute_seq_tours"]["demographics_path"]

    # input_metrics_path = cfg["paths"]["compute_mobility_metrics"]["output_metrics_path"]

    output_path_seqeff = cfg["paths"]["compute_seq_tours"]["output_seqeff_path"]
    output_path_tours = cfg["paths"]["compute_seq_tours"]["output_tour_path"]

    output_dffinal_path = cfg["paths"]["compute_seq_tours"]["output_dffinal_path"]

    output_metrics_fname = cfg["paths"]["compute_mobility_metrics"]["output_filename"]
    output_seqeff_fname = cfg["paths"]["compute_seq_tours"]["output_seqeff_filename"]
    output_tour_fname = cfg["paths"]["compute_seq_tours"]["output_tour_filename"]

    # ---- Initialize Spark ----
    spark = start_spark(
        n_workers=job_cfg["n_workers"],
        temp_folder=job_cfg["temp_folder"],
        mem=job_cfg["mem"],
    )

    # ---- Run by user groups ----
    if args.user_group is not None:
        # USER SPECIFIED A SINGLE GROUP
        user_groups = [args.user_group]
        print(f"> Running only for: {args.user_group}")

    else:
        # RUN FOR ALL GROUPS
        user_groups = list_user_groups(input_path)
        print(f"> Found {len(user_groups)} user groups.")

    if seqtours_cfg["get_efficiency"]:
        print(f"\n>> Get sequence efficiency metrics including Home anchor")

    for ug in tqdm(user_groups, desc="Processing user groups"):
        time_ug_start = time.time()

        print(f"\n>> Get tours for {ug}")
        sdf_stops = spark.read.load(os.path.join(input_path, ug))

        # ---- Preprocessing ----
        sdf_stops = sdf_stops.filter((F.col("loc").isNotNull()) & (F.col("loc") != -1))

        # Transform to local time and selected temporal resolution ----
        sdf_stops_lt = local_time_resolution(
            sdf_stops, time_agg_cols=agg_cfg["time_agg_cols"]
        )

        # [Just for validation] Prepare data to analyze efficiency from Home anchor----
        if seqtours_cfg["get_efficiency"]:

            # Get Home/work locations using HoWDe
            howde_params = howde_cfg["params_config"]
            sdf_hw = HoWDe_labelling(sdf_stops_lt, edit_config_default=howde_params)
            sdf_stops_lt = sdf_stops_lt.join(
                sdf_hw.select(
                    "useruuid",
                    "start",
                    "end",
                    "loc",
                    "location_type",
                    "detect_H_loc",
                    "detect_W_loc",
                ),
                on=["useruuid", "start", "end", "loc"],
                how="left",
            )

            # Update location id with location type (home/work)
            sdf_stops_h = (
                sdf_stops_lt.groupBy(
                    "useruuid", *agg_cfg["time_agg_cols"], "detect_H_loc"
                )
                .count()
                .filter(F.col("detect_H_loc").isNotNull())
                .orderBy("count", ascending=False)
                .groupBy("useruuid", *agg_cfg["time_agg_cols"])
                .agg(F.first("detect_H_loc").alias("home_loc_M"))
            )

            sdf_stops_w = (
                sdf_stops_lt.groupBy(
                    "useruuid", *agg_cfg["time_agg_cols"], "detect_W_loc"
                )
                .count()
                .filter(F.col("detect_W_loc").isNotNull())
                .orderBy("count", ascending=False)
                .groupBy("useruuid", *agg_cfg["time_agg_cols"])
                .agg(F.first("detect_W_loc").alias("work_loc_M"))
            )

            sdf_stops_hw = (
                sdf_stops_lt.join(
                    sdf_stops_h, on=["useruuid", *agg_cfg["time_agg_cols"]], how="left"
                )
                .join(
                    sdf_stops_w, on=["useruuid", *agg_cfg["time_agg_cols"]], how="left"
                )
                .withColumn(
                    "stops",
                    F.when((F.col("home_loc_M") == F.col("loc")), F.lit("H"))
                    .when((F.col("work_loc_M") == F.col("loc")), F.lit("W"))
                    .otherwise(F.col("loc")),
                )
            ).orderBy(["useruuid", "start_month"])

            if job_cfg["verbose"]:
                print(">> Home, work detected loc updated:")

        # Join demographics
        sdf_demo = spark.read.load(demo_path)
        sdf_stops_lt = sdf_stops_lt.join(
            sdf_demo.select(["useruuid"] + agg_cfg["demo_cols"]),
            on="useruuid",
            how="left",
        ).filter(F.col("gender").isNotNull())

        if job_cfg["verbose"]:
            print(">> Demographics added")

        # ---- Get stop sequences per user and temporal resolution ----
        sdf_stops_seq = get_stop_sequences(
            sdf_stops_lt, agg_cfg["base_cols"], agg_cfg["time_agg_cols"]
        )
        sdf_stops_seq = sdf_stops_seq.withColumn(
            "tmp",
            udf_clean_seq_consec_stops(
                "stop_sequence", "positions", "arr_time", "dep_time"
            ),
        ).select(
            *agg_cfg["base_cols"],
            *agg_cfg["time_agg_cols"],
            F.col("tmp.stop_sequence").alias("stop_sequence"),
            F.col("tmp.stop_time").alias("stop_time"),
            F.col("tmp.arr_time").alias("arr_time"),
            F.col("tmp.dep_time").alias("dep_time"),
            F.col("tmp.positions").alias("positions"),
            F.col("tmp.seq_len").alias("seq_len"),
        )

        if job_cfg["verbose"]:
            print(">> Stops sequences computed and cleaned")

        # ---- Get tours and keystones ----
        sdf_tours = (
            sdf_stops_seq.withColumn(
                "tmp",
                udf_extract_tours_and_keystones(
                    "stop_sequence", "arr_time", "dep_time", "positions"
                ),
            )
            .select(
                "*",
                "tmp.filtered_tours",
                "tmp.keystones",
            )
            .drop(*["tmp"])
            .withColumn(
                "main_seq_keystone",
                udf_get_main_keystone(
                    "keystones", "stop_sequence", "arr_time", "dep_time"
                ),
            )
        )
        if job_cfg["verbose"]:
            print(">> Tours and keystones computed")

        # ---- [Just for validation] Get efficiency metrics for sequences with Home anchor
        sdf_results = sdf_tours.select("*")

        if seqtours_cfg["get_efficiency"]:
            sdf_results = (
                sdf_results.withColumn(
                    "tmp",
                    udf_get_seq_distances(
                        "stop_sequence", "positions", "main_seq_keystone"
                    ),
                )
                .select(
                    "*",
                    "tmp.seq_distances",
                    "tmp.seq_cost",
                    "tmp.seq_dist_from_start",
                    "tmp.seq_dist_from_home",
                    "tmp.seq_dist_from_keystone",
                    "tmp.seq_reward_start",
                    "tmp.seq_efficency_start",
                    "tmp.seq_savings_start",
                    "tmp.seq_reward_home",
                    "tmp.seq_efficency_home",
                    "tmp.seq_savings_home",
                    "tmp.seq_reward_keystone",
                    "tmp.seq_efficency_keystone",
                    "tmp.seq_savings_keystone",
                )
                .drop(*["tmp"])
            )

            if job_cfg["verbose"]:
                print(">> Sequence efficiency metrics added")

        # ---- Get journey metrics ----
        sdf_tours = add_jrny_reward_columns(sdf_results)
        sdf_tour_results = (
            sdf_tours.withColumn("tmp", udf_summarize_tours("tours"))
            .select(
                "*",
                "tmp.jrny_total",
                "tmp.jrny_total_unique",
                "tmp.jrny_lists",
                "tmp.jrny_length",
                "tmp.jrny_cost",
                "tmp.jrny_reward",
                "tmp.jrny_max_reward",
                "tmp.jrny_duration",
            )
            .drop("tmp")
        )

        if job_cfg["verbose"]:
            print(">> Tours metrics added")

        # ---- [INTERNAL] Save seq_eff and journey dataset ----
        # select key columns

        sdf_tour_results.write.mode("overwrite").parquet(
            os.path.join(output_path_seqeff, ug)
        )
        time_tot = round((time.time() - time_start) / 60, 2)
        print(
            f">> Saved {ug} spark seq_eff at {output_path_seqeff} (total: {time_tot}min)"
        )

        # ---- Get pandas df:
        df_ug = (
            sdf_tour_results.filter(F.size("jrny_lists") > 0)
            .select(
                *agg_cfg["base_cols"],
                *agg_cfg["time_agg_cols"],
                "jrny_lists",
                "jrny_length",
                "jrny_cost",
                "jrny_reward",
                "jrny_max_reward",
                "jrny_duration",
            )
            .toPandas()
        )

        # ---- Explode  pandas df:
        df_tours_exp = df_ug.explode(
            [
                "jrny_lists",
                "jrny_length",
                "jrny_cost",
                "jrny_reward",
                "jrny_max_reward",
                "jrny_duration",
            ]
        )

        # --- Get length bins for interpretation and visualization:
        df_tours_exp = get_length_bin(df_tours_exp)

        # ---- Save ug exploded journey dataset:
        df_tours_exp.to_csv(os.path.join(output_path_tours, ug), index=False)
        time_tot = round((time.time() - time_start) / 60, 2)
        print(f">> Saved {ug} df tours at {output_path_tours} (total: {time_tot}min)")

    time_tot = round((time.time() - time_start) / 60, 2)
    print(f">> End ug iterations (total: {time_tot}min)")

    # ---- Save final exploded journey dataset:
    # Open exploded jrny files and save as CSV (all ug)
    df_tours_all = pd.concat(
        [pd.read_csv(os.path.join(output_path_tours, f"{ug}")) for ug in user_groups],
        ignore_index=True,
    )

    # Add activity/repertoire deciles split
    fname = os.path.join(output_dffinal_path, output_metrics_fname)
    df_metrics = pd.read_csv(fname)
    df_tours_all = df_tours_all.merge(
        df_metrics[
            ["useruuid", "start_month", "activity_repertoire_groups"]
        ].drop_duplicates(),
        on=["useruuid", "start_month"],
        how="left",
    ).rename(
        columns={
            "jrny_lists": "tour",
            "jrny_length": "tour_total_stops",
            "jrny_cost": "tour_total_distance",
            "jrny_reward": "tour_total_reward",
            "jrny_max_reward": "tour_max_reward",
            "jrny_duration": "tour_total_duration",
        }
    )

    print(f">> Merged activity_repertoire_groups from metrics to tours dataset")

    fname = os.path.join(output_dffinal_path, output_tour_fname)
    df_tours_all.to_csv(fname, index=False)
    time_tot = round((time.time() - time_start) / 60, 2)
    print(f">> Saved all at {fname} (total: {time_tot}min)")

    # Open seq_eff file and save as CSV
    if seqtours_cfg["get_efficiency"]:
        df_seqeff_all = pd.concat(
            [
                spark.read.parquet(os.path.join(output_path_seqeff, f"{ug}")).toPandas()
                for ug in user_groups
            ],
            ignore_index=True,
        )

        fname = os.path.join(output_dffinal_path, output_seqeff_fname)
        df_seqeff_all.to_csv(fname, index=False)
        time_tot = round((time.time() - time_start) / 60, 2)
        print(f">> Saved all at {fname} (total: {time_tot}min)")


if __name__ == "__main__":
    main()
