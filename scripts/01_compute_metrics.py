"""
Compute mobility and network metrics from stop-level mobility data.

This script:
1. Loads stop data partitioned by user groups.
2. Converts timestamps to local time and applies temporal aggregation.
3. Optionally detects Home/Work locations using HoWDe.
4. Joins demographic attributes.
5. Computes mobility metrics via MobilityMetricsPipeline.
6. Builds mobility networks and if configured, stores OD pairs
7. Computes network metrics via NetworkMetricsPipeline.
8. Joins all metrics, applies anonymization if configured, and writes results.

Execution is controlled via a YAML configuration file and can be run
for all user groups or a single specified group.

Typical usage:
    python ./scripts/01_compute_metrics.py --config ./configs/monthly_met_all_config.yaml
    python ./scripts/01_compute_metrics.py --config ./configs/monthly_met_all_config.yaml --user_group user_group=00
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
from compute_metrics.io import list_user_groups, load_config, local_time_resolution
from compute_metrics.pipelines.mobility_metrics_pipeline import MobilityMetricsPipeline
from compute_metrics.pipelines.network_metrics_pipeline import NetworkMetricsPipeline


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
    howde_cfg = cfg["home_work_detection"]
    met_cfg = cfg["metrics"]
    anonym_cfg = cfg["anonymization"]

    input_path = cfg["paths"]["compute_mobility_metrics"]["input_path"]
    demo_path = cfg["paths"]["compute_mobility_metrics"]["demographics_path"]
    output_path_metrics = cfg["paths"]["compute_mobility_metrics"][
        "output_metrics_path"
    ]
    output_networks_path = cfg["paths"]["compute_mobility_metrics"][
        "output_networks_path"
    ]
    output_allmet_path = cfg["paths"]["compute_mobility_metrics"][
        "output_all_metrics_path"
    ]
    output_fname = cfg["paths"]["compute_mobility_metrics"]["output_filename"]

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

    print(
        f"\n>> Get {len(met_cfg['mobility_metrics'])} mobility metrics: {met_cfg['mobility_metrics']}"
    )
    print(
        f"\n>> Get {len(met_cfg['mobility_aux_metrics'])} mobility auxiliary metrics: {met_cfg['mobility_aux_metrics']}"
    )
    print(
        f"\n>> Get {len(met_cfg['network_metrics'])} network metrics: {met_cfg['network_metrics']}"
    )
    print(
        f"\n>> Get {len(met_cfg['efficiency_metrics'])} efficiency metrics: {met_cfg['efficiency_metrics']}"
    )

    for ug in tqdm(user_groups, desc="Processing user groups"):
        time_ug_start = time.time()

        print(f"\n>> Running mobility metrics for {ug}")
        sdf_stops = spark.read.load(os.path.join(input_path, ug))

        # ---- Preprocessing ----
        sdf_stops = sdf_stops.filter((F.col("loc").isNotNull()) & (F.col("loc") != -1))

        # Transform to local time and selected temporal resolution ----
        sdf_stops_lt = local_time_resolution(
            sdf_stops, time_agg_cols=agg_cfg["time_agg_cols"]
        )

        # Get Home/Work locations
        if howde_cfg["get_howde"]:
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

        # Join demographics
        sdf_demo = spark.read.load(demo_path)
        sdf_stops_lt = sdf_stops_lt.join(
            sdf_demo.select(["useruuid"] + agg_cfg["demo_cols"]),
            on="useruuid",
            how="left",
        ).filter(F.col("gender").isNotNull())

        # Sort
        COLS = [
            "useruuid",
            *agg_cfg["demo_cols"],
            *agg_cfg["time_agg_cols"],
            "start",
            "end",
            "loc",
            "latitude",
            "longitude",
        ]
        if howde_cfg["get_howde"]:
            COLS += ["location_type", "detect_H_loc", "detect_W_loc"]

        # print(COLS)
        sdf_stops_lt = sdf_stops_lt.select(*COLS).orderBy(["useruuid", "start"])

        # Cache the sdf_stops_lt before running metrics
        stops_lt_hw = sdf_stops_lt.cache()
        stops_lt_hw.count()  # MATERIALIZE cache

        # ---- Run Mobility Metrics Pipeline ----
        mobmet_pipeline = MobilityMetricsPipeline(
            sdf=stops_lt_hw,
            agg_cfg=agg_cfg,
            met_cfg=met_cfg,
        )

        sdf_mob = mobmet_pipeline.run()

        # Filter to a min visits and time recorded in stops
        # sdf_mob = sdf_mob.filter((F.col("locations") >= F.lit(2)) & (F.col("total_stay_time") >= F.lit(10))) # -> Do we want to filter?
        sdf_mob = sdf_mob.filter((F.col("locations") >= F.lit(2)))

        if job_cfg.get("verbose", False):
            time_ug = round((time.time() - time_ug_start) / 60, 2)
            time_tot = round((time.time() - time_start) / 60, 2)
            print(
                f">> Mobility Metrics complete for {ug} (ug time: {time_ug}min, total: {time_tot}min)"
            )

        # ---- Run Network Metrics Pipeline ----
        netmet_pipeline = NetworkMetricsPipeline(
            sdf=stops_lt_hw,
            agg_cfg=agg_cfg,
            met_cfg=met_cfg,
            out_netw_dir=os.path.join(output_networks_path, ug),
        )

        sdf_net = netmet_pipeline.run()

        if job_cfg.get("verbose", False):
            time_ug = round((time.time() - time_ug_start) / 60, 2)
            time_tot = round((time.time() - time_start) / 60, 2)
            print(
                f">> Network Metrics complete for {ug} (ug time: {time_ug}min, total: {time_tot}min)"
            )

        # ---- Join all Mobility and Network Metrics ----
        sdf_met = sdf_mob.join(
            sdf_net, on=agg_cfg["base_cols"] + agg_cfg["time_agg_cols"], how="left"
        )

        # ---- Write Output ----
        out_dir = os.path.join(output_path_metrics, ug)
        # print(f">> Saving to {out_dir}")
        sdf_met.write.mode("overwrite").parquet(
            out_dir
        )  ## SAVE AS CSV? ALSO NETWORKS OD FILES?
        sdf_stops_lt.unpersist()

        time_ug = round((time.time() - time_ug_start) / 60, 2)
        time_tot = round((time.time() - time_start) / 60, 2)
        print(f">> Saved {ug} in {time_ug}min (total: {time_tot}min)")

    # ---- Get Pandas Dataframe across all saved ug Output ----
    if agg_cfg.get("get_allmet_pdf", False):
        # all_ugs = [f for f in os.listdir(output_path_metrics) if os.path.isdir(os.path.join(output_path_metrics, f))]
        all_dfs = []
        for ug in user_groups:
            df = spark.read.parquet(os.path.join(output_path_metrics, ug)).toPandas()
            df["user_group"] = ug
            all_dfs.append(df)

        final_df = pd.concat(all_dfs, ignore_index=True)

        # Add Activty/repertoire deciles split
        final_df["activity_deciles"] = (
            pd.qcut(
                final_df["visits"],
                q=10,
                labels=False,  # gives 0–9 integers
                duplicates="drop",  # avoids crash if ties exist
            )
            + 1
        )  # updates to 1–10 instead of 0–9

        final_df["repertoire_deciles"] = (
            pd.qcut(
                final_df["locations"],
                q=10,
                labels=False,  # gives 0–9 integers
                duplicates="drop",  # avoids crash if ties exist
            )
            + 1
        )  # updates to 1–10 instead of 0–9

        final_df["activity_repertoire_groups"] = final_df.apply(
            lambda row: (
                "inactive"
                if row["activity_deciles"] <= 3 and row["repertoire_deciles"] <= 3
                else (
                    "moderate"
                    if row["activity_deciles"] > 3
                    and row["activity_deciles"] <= 7
                    and row["repertoire_deciles"] > 3
                    and row["repertoire_deciles"] <= 7
                    else (
                        "active"
                        if row["activity_deciles"] >= 8
                        and row["repertoire_deciles"] >= 8
                        else "other"
                    )
                )
            ),
            axis=1,
        )

        # Save as csv
        final_df.to_csv(os.path.join(output_allmet_path, output_fname), index=False)
        time_tot = round((time.time() - time_start) / 60, 2)
        print(
            f">> Saved all at {output_allmet_path + output_fname} (total: {time_tot}min)"
        )

    spark.stop()


if __name__ == "__main__":
    main()

# --- HOW TO RUN ---
#  python ./scripts/01_compute_metrics.py --config ./configs/monthly_met_config.yaml
#  python ./scripts/01_compute_metrics.py --config ./configs/monthly_met_config.yaml --user_group user_group=00
