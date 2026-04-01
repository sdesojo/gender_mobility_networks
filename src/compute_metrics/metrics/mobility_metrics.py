# src/mobmet/metrics/metrics.py

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from typing import Sequence

from utils.utils import distance


def visits(
    sdf: DataFrame,
    base_cols: Sequence[str],
    time_sort_cols: Sequence[str],
    time_agg_cols: Sequence[str],
    suffix: str = "",
) -> DataFrame:

    group_cols = list(base_cols) + list(time_agg_cols)
    w = Window.partitionBy(*group_cols)

    return sdf.withColumn(f"visits{suffix}", F.count("loc").over(w))


def locations(
    sdf: DataFrame,
    base_cols: Sequence[str],
    time_sort_cols: Sequence[str],
    time_agg_cols: Sequence[str],
    suffix: str = "",
) -> DataFrame:
    group_cols = list(base_cols) + list(time_agg_cols)

    w_rank = Window.partitionBy(*group_cols).orderBy("loc")
    w_max = Window.partitionBy(*group_cols)

    return (
        sdf.withColumn("_loc_rank", F.dense_rank().over(w_rank))
        .withColumn(f"locations{suffix}", F.max("_loc_rank").over(w_max))
        .drop("_loc_rank")
    )


# -- DISTANCE-BASED METRICS ----------------------------------------
def total_cost_travel_distance(
    sdf: DataFrame,
    base_cols: Sequence[str],
    time_sort_cols: Sequence[str],
    time_agg_cols: Sequence[str],
    suffix: str = "",
) -> DataFrame:
    """Compute total distance travelled (travel cost) in km per (base_cols + time_agg_cols) group and attach it as a column.

    Expects sdf to have:
    - 'latitude', 'longitude' : coordinates of each stop
    """
    main_cols = base_cols + time_agg_cols

    w = Window.partitionBy(*main_cols).orderBy(*time_sort_cols)
    w_loc = Window.partitionBy(*main_cols + ["loc"]).orderBy(*time_sort_cols)

    # Keep one lat-lon per location by taking the median
    sdf_ = sdf.withColumn(
        "latitude", F.percentile_approx("latitude", 0.5, 10000).over(w_loc)
    ).withColumn("longitude", F.percentile_approx("longitude", 0.5, 10000).over(w_loc))

    # Get distance between consecutive stops
    sdf_with_dist = (
        sdf_.withColumn("prev_lat", F.lag("latitude").over(w))
        .withColumn("prev_lon", F.lag("longitude").over(w))
        .withColumn(
            "step_distance",
            distance(
                F.col("latitude"),
                F.col("longitude"),
                F.col("prev_lat"),
                F.col("prev_lon"),
            ),
        )
        .withColumn(
            "_next_loc",
            F.when(F.lag("loc").over(w).isNotNull(), F.lag("loc").over(w)).otherwise(
                "first_loc"
            ),
        )
        .withColumn(
            "is_self_loop", F.when(F.col("loc") == F.col("_next_loc"), 1).otherwise(0)
        )
        .filter(F.col("step_distance").isNotNull() & (F.col("is_self_loop") == 0))
    )

    # Sum step distances per month (km)
    tot_dist_per_bucket = sdf_with_dist.groupBy(*main_cols).agg(
        F.round(F.sum("step_distance"), 5).alias(f"total_cost_travel_distance{suffix}")
    )

    # Filter to realistic distances
    colname = f"total_cost_travel_distance{suffix}"
    tot_dist_per_bucket = tot_dist_per_bucket.withColumn(
        colname,
        F.when(F.col(colname).isNull(), None)
        .when(F.col(colname) >= 0.1, F.col(colname))
        .otherwise(0),
    )

    # Join the metric back to the original sdf
    sdf_tot_dist = sdf.join(tot_dist_per_bucket, on=main_cols, how="left")

    return sdf_tot_dist


def total_reward_home_distance(
    sdf: DataFrame,
    base_cols: Sequence[str],
    time_sort_cols: Sequence[str],
    time_agg_cols: Sequence[str],
    suffix: str = "",
) -> DataFrame:
    """Compute total distance between home and all locations (travel reward) in km per (base_cols + time_agg_cols) group and attach it as a column.

    Expects sdf to have:
    - 'latitude', 'longitude' : coordinates of each stop
    - 'location_type", 'detect_H_loc': home location label
    """

    main_cols = base_cols + time_agg_cols
    home_cols = ["home_loc", "home_lat", "home_lon"]

    w_h = Window.partitionBy(*main_cols + ["loc"])
    sdf_home = (
        sdf.filter(F.col("location_type") == "H")
        .withColumn(
            f"total_home_time",
            F.round(F.sum((F.col("end") - F.col("start"))).over(w_h)),
        )
        .select(main_cols + ["loc", "latitude", "longitude", "total_home_time"])
        .dropDuplicates()
        # Keep one lat-lon per home location by taking the median
        .groupBy(main_cols + ["loc"])
        .agg(
            F.percentile_approx("latitude", 0.5, 10000).alias("home_lat"),
            F.percentile_approx("longitude", 0.5, 10000).alias("home_lon"),
            F.sum("total_home_time").alias("total_home_time"),
        )
        .withColumnRenamed("loc", "home_loc")
    )

    # Select the home anchor as the detected home in a month where most time was spent [only one lat-lon per home]
    w_ht = Window.partitionBy(*main_cols).orderBy(F.desc("total_home_time"))
    sdf_top_home = (
        sdf_home.withColumn("rn", F.row_number().over(w_ht))
        .where(F.col("rn") == 1)
        .select(*main_cols, *home_cols)
    )

    # Join the home anchor back to the original sdf and compute distance of each stop to home
    w = Window.partitionBy(*main_cols).orderBy(*time_sort_cols)
    sdf_with_dist = (
        sdf.join(sdf_top_home, on=main_cols, how="left")
        .withColumn(
            "home_distance",
            distance(
                F.col("latitude"),
                F.col("longitude"),
                F.col("home_lat"),
                F.col("home_lon"),
            ),
        )
        # .withColumn(
        #     "is_valid_step",
        #     F.when(F.lag("loc").over(w) != F.col("loc"), 1).otherwise(0)
        # )
        # .filter(F.col("home_distance").isNotNull() & (F.col("is_valid_step") == 1))
        .withColumn(
            "is_home_loc", F.when(F.col("loc") == F.col("home_loc"), 1).otherwise(0)
        )
        .withColumn(
            "_next_loc",
            F.when(F.lag("loc").over(w).isNotNull(), F.lag("loc").over(w)).otherwise(
                "first_loc"
            ),
        )
        .withColumn(
            "is_self_loop", F.when(F.col("loc") == F.col("_next_loc"), 1).otherwise(0)
        )
        .filter(
            F.col("home_distance").isNotNull()
            & (F.col("is_home_loc") == 0)
            & (F.col("is_self_loop") == 0)
        )
    )

    # Sum step distances per month (km)
    tot_dist_per_bucket = sdf_with_dist.groupBy(*main_cols).agg(
        F.first("home_loc").alias(f"home_loc"),
        F.round(F.sum("home_distance"), 5).alias(f"sum_home_distance{suffix}"),
        F.round(2 * F.sum("home_distance"), 5).alias(
            f"total_reward_home_distance{suffix}"
        ),
        F.round(F.max("home_distance"), 5).alias(f"max_home_distance{suffix}"),
        F.round(2 * F.max("home_distance"), 5).alias(
            f"max_reward_home_distance{suffix}"
        ),
    )

    # Join the metric back to the original sdf
    sdf_tot_dist = (
        sdf.join(tot_dist_per_bucket, on=main_cols, how="left")
        .withColumn("has_home", F.when(F.col("home_loc").isNotNull(), 1).otherwise(0))
        .drop("home_loc")
    )

    return sdf_tot_dist


def tour_efficiency_dist(
    sdf: DataFrame,
    base_cols: Sequence[str],
    time_sort_cols: Sequence[str],
    time_agg_cols: Sequence[str],
    suffix: str = "",
) -> DataFrame:
    """Compute tour efficiency (reward/travel cost) in km per (base_cols + time_agg_cols) group and attach it as a column.

    Expects sdf to have the distance-based metrics already computed and attached as columns.
    """
    main_cols = base_cols + time_agg_cols

    # test if the required distance metrics are present
    required_cols = [
        f"total_reward_home_distance{suffix}",
        f"total_cost_travel_distance{suffix}",
    ]

    for col in required_cols:
        if col not in sdf.columns:
            print(f"Warning: Tour efficiency not computed since {col} not found.")
            return sdf.withColumn(
                f"tour_efficiency_dist{suffix}",
                F.lit(None),
            )

    reward = F.col(f"total_reward_home_distance{suffix}")
    cost = F.col(f"total_cost_travel_distance{suffix}")

    sdf_eff = sdf.withColumn(
        f"tour_efficiency_dist{suffix}",
        F.when(
            ((reward > 0) & (cost > 0) & (reward >= cost)),  # guard
            1 - (F.log10(cost) / F.log10(reward)),
        ).otherwise(None),
    )
    return sdf_eff


# -- TIME-BASED METRICS ----------------------------------------
def total_stay_time(
    sdf: DataFrame,
    base_cols: Sequence[str],
    time_sort_cols: Sequence[str],
    time_agg_cols: Sequence[str],
    suffix: str = "",
) -> DataFrame:
    """Compute total stop time in hours per (base_cols + time_agg_cols) group and attach it as a column.

    Expects sdf to have:
    - 'start' and 'end' unix timestamp columns for each stop
    """
    main_cols = base_cols + time_agg_cols

    w = Window.partitionBy(*main_cols)

    # Sum stop durations (in sec)
    sdf_total_time = sdf.withColumn(
        f"total_stay_time{suffix}",
        F.round(F.sum((F.col("end") - F.col("start"))).over(w)),
    )

    # Get Temporal Coverage
    sdf_total_time = (
        sdf_total_time.withColumn(
            "total_days_M", F.dayofmonth(F.last_day(F.from_unixtime("start")))
        ).withColumn(
            "coverage_stay_time",
            F.col(f"total_stay_time{suffix}")
            / (F.lit(24 * 3600) * F.col("total_days_M")),
        )
    ).drop("total_days_M")

    return sdf_total_time


def total_travel_time(
    sdf: DataFrame,
    base_cols: Sequence[str],
    time_sort_cols: Sequence[str],
    time_agg_cols: Sequence[str],
    suffix: str = "",
) -> DataFrame:
    """Compute total distance travelled in km per (base_cols + time_agg_cols) group and attach it as a column.

    Expects sdf to have:
    - 'latitude', 'longitude' : coordinates of each stop
    """
    main_cols = base_cols + time_agg_cols

    w = Window.partitionBy(*main_cols).orderBy(*time_sort_cols)
    # w_ud = Window.partitionBy("useruuid", 'date').orderBy(*time_sort_cols)
    MAX_TIME = 12 * 3600

    sdf_with_dist = (
        sdf
        # .withColumn('date', F.to_date(F.from_unixtime(F.col('start'))))
        # .withColumn('start_travel', F.lag('end').over(w_ud))
        .withColumn("start_travel", F.lag("end").over(w))
        # Define Trip Time
        .withColumn("travel_time", F.col("start") - F.col("start_travel"))
        # Filter to sensical travel time
        .filter(
            (F.col("travel_time").isNotNull())
            & (F.col("travel_time") >= F.lit(0))
            & (F.col("travel_time") < F.lit(MAX_TIME))
        )
    )

    # Sum travel time (sec)
    tot_dist_per_bucket = sdf_with_dist.groupBy(*main_cols).agg(
        F.round(F.sum("travel_time"), 3).alias(f"total_travel_time{suffix}")
    )

    # Join the metric back to the original sdf
    sdf_tot_dist = sdf.join(tot_dist_per_bucket, on=main_cols, how="left")

    return sdf_tot_dist


# def home_travel_time(
#     sdf: DataFrame,
#     base_cols: Sequence[str],
#     time_sort_cols: Sequence[str],
#     time_agg_cols: Sequence[str],
#     suffix: str = "",
# ) -> DataFrame:

#     # .alias(f"REWARD_total_travel_time{suffix}"
#     return None
