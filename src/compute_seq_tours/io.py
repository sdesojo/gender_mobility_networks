# from __future__ import annotations
from typing import List, Any, Dict
import os
import yaml

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import *


def list_user_groups(path: str):
    """Return all folders like 'user_group=00', 'user_group=01', ..."""
    if not os.path.exists(path):
        return []
    return sorted([d for d in os.listdir(path) if d.startswith("user_group=")])


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a single YAML config file (e.g. configs/agg_daily.yaml).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ---- Preprocessing ----


# -- Update to local time and selected temporal resolution
def local_time_resolution(df: DataFrame, time_agg_cols: List[str]) -> DataFrame:

    # Precompute local-time seconds since epoch
    local_secs = F.col("start") + F.col("timezone")
    DAY = 24 * 3600
    WEEK = 7 * DAY

    if "start_day" in time_agg_cols:
        # Local-epoch timestamp for midnight of that day
        df = df.withColumn("start_day", F.floor(local_secs / DAY) * DAY)

    if "start_week" in time_agg_cols:
        # Local-epoch timestamp for the start of the local week
        df = df.withColumn(
            "start_week", F.floor((local_secs - 4 * DAY) / WEEK) * WEEK + 4 * DAY
        )

    if "start_hourbin_week" in time_agg_cols:
        # Define Hour bin: [0,4,8,12,16,20]
        df = (
            df.withColumn("start_hour", F.hour(F.from_unixtime(local_secs)))
            .withColumn(
                "start_hour_bin", (F.floor(F.col("start_hour") / 4) * 4).cast("int")
            )
            .drop("start_hour")
        )

        # Weekday / Weekend [Spark: dayofweek: 1=Sun, ..., 7=Sat]
        df = df.withColumn(
            "weekday_weekend",
            F.when(
                F.dayofweek(F.from_unixtime(local_secs)).isin([1, 7]),
                F.lit("weekend"),
            ).otherwise(F.lit("weekday")),
        )

        # Local-epoch timestamp for the start of the local week
        df = df.withColumn(
            "start_week", F.floor((local_secs - 4 * DAY) / WEEK) * WEEK + 4 * DAY
        )

    if "start_month" in time_agg_cols:
        # Local-epoch timestamp for the start of the local month
        # consider calendar months have differing number of days
        df = df.withColumn(
            "start_local_ts", F.from_unixtime(local_secs).cast("timestamp")
        )
        df = df.withColumn(
            "start_local_month_ts", F.date_trunc("month", F.col("start_local_ts"))
        )

        df = df.withColumn(
            "start_month", F.unix_timestamp("start_local_month_ts")
        ).drop("start_local_ts", "start_local_month_ts")

    # Keep local time only
    df = df.withColumn("start", local_secs).withColumn(
        "end", F.col("end") + F.col("timezone")
    )

    return df


# -- Post-processing
