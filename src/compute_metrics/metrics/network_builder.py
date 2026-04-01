from __future__ import annotations

from typing import Sequence, Tuple

from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F

from utils.utils import distance


## -- Get Networks from stops dataframe ----------------------------------------
def get_nodes_edges(
    sdf: DataFrame,
    base_cols: Sequence[str],
    time_sort_cols: Sequence[str],
    time_agg_cols: Sequence[str],
    loc_col: str = "loc",
    suffix: str = "",
) -> tuple[DataFrame, DataFrame]:
    """
    Build weighted undirected edges + nodes per (base_cols + time_agg_cols). Keep the home location per month
    - edges: (u, v) sorted, with weight = transitions count
    - nodes: node weight = visits count (excluding self-loops if you want; here includes all visits that participate in transitions)
    """
    main_cols = list(base_cols) + list(time_agg_cols)
    w = Window.partitionBy(*main_cols).orderBy(*time_sort_cols)

    # Pair consecutive locations:
    sdf_pairs = (
        sdf.withColumn("_next_loc", F.lead(F.col(loc_col), 1).over(w))
        .withColumn("_next_lat", F.lead("latitude", 1).over(w))
        .withColumn("_next_lon", F.lead("longitude", 1).over(w))
        .where(F.col(loc_col).isNotNull())
        # Drop self-loops (same location next) but keep transitions to missing next (e.g., last stop in month)
        .where(F.col(loc_col) != F.col("_next_loc"))
    )

    # Select the home anchor as the detected home in a month where most time was spent
    w_h = Window.partitionBy(*main_cols + ["loc"])
    w_ht = Window.partitionBy(*main_cols).orderBy(F.desc("total_home_time"))

    sdf_top_home = (
        sdf_pairs.filter(F.col("location_type") == "H")
        .withColumn(
            f"total_home_time",
            F.round(F.sum((F.col("end") - F.col("start"))).over(w_h)),
        )
        .select(main_cols + ["loc", "total_home_time"])  # "latitude", "longitude",
        .dropDuplicates()
        .withColumnRenamed("loc", "home_loc")
        # .withColumnRenamed("latitude", "home_lat")
        # .withColumnRenamed("longitude", "home_lon")
        .withColumn("rn", F.row_number().over(w_ht))
        .where(F.col("rn") == 1)
        .select(*main_cols + ["home_loc"])
    )

    # Join the home anchor back to the original sdf with undirected edges
    w = Window.partitionBy(*main_cols).orderBy(*time_sort_cols)
    sdf_pairs = sdf_pairs.join(sdf_top_home, on=main_cols, how="left").withColumn(
        "home_loc",
        F.when(F.col("home_loc").isNull(), F.lit("missing_home")).otherwise(
            F.col("home_loc")
        ),
    )

    # Get nodes dataframe, weight = count of visits to each location (not self-loops) per main_cols + home_loc.
    sdf_nodes = (
        sdf_pairs.groupBy(
            *(
                main_cols
                + ["home_loc", F.col(loc_col).cast("string").alias(f"loc{suffix}")]
            )
        )
        .agg(F.count(F.lit(1)).alias(f"node_weight_und{suffix}"))
        .select(*(main_cols + ["home_loc", f"loc{suffix}", f"node_weight_und{suffix}"]))
    )

    # Get pairs, drop last stop (no next)
    sdf_pairs = sdf_pairs.where(F.col("_next_loc").isNotNull()).withColumn(
        "_und_edge",
        F.sort_array(
            F.array(F.col(loc_col).cast("int"), F.col("_next_loc").cast("int"))
        ),
    )

    # Get undirected network edges dataframe, weight = count of transitions for each undirected edge (u,v) per main_cols + home_loc.
    sdf_edges = (
        sdf_pairs.groupBy(*(main_cols + ["home_loc", "_und_edge"]))
        .agg(F.count(F.lit(1)).alias(f"edge_weight_und{suffix}"))
        .select(
            *(
                main_cols
                + [
                    "home_loc",
                    F.col("_und_edge").alias(f"edges_pairs_undirected{suffix}"),
                    F.col(f"edge_weight_und{suffix}"),
                ]
            )
        )
    )

    return sdf_nodes, sdf_edges, sdf_pairs


def compress_nodes_edges_to_maps(
    sdf_nodes: DataFrame,
    sdf_edges: DataFrame,
    base_cols: Sequence[str],
    time_sort_cols: Sequence[str],
    time_agg_cols: Sequence[str],
    loc_col: str,
    node_w_col: str,
    edge_pair_col: str,
    edge_w_col: str,
    edge_delim: str = "-",
) -> DataFrame:
    """
    Pure Spark compression:
    - nodes_map: map<str,int> where key=node, value=node_weight
    - edges_map: map<str,int> where key="u-v" (undirected edges), value=edge_weight
    """
    main_cols = list(base_cols) + list(time_agg_cols)

    # nodes: map_from_entries(collect_list(struct(key, value)))
    nodes_map_df = sdf_nodes.groupBy(*main_cols + ["home_loc"]).agg(
        F.map_from_entries(
            F.collect_list(F.struct(F.col(loc_col), F.col(node_w_col)))
        ).alias("nodes_weights")
    )

    # edges: array<string> -> "u-v"
    edges_map_df = (
        sdf_edges.withColumn(
            "_edge_key",
            F.concat_ws(edge_delim, F.col(edge_pair_col)[0], F.col(edge_pair_col)[1]),
        )
        .groupBy(*main_cols + ["home_loc"])
        .agg(
            F.map_from_entries(
                F.collect_list(F.struct(F.col("_edge_key"), F.col(edge_w_col)))
            ).alias("edges_weights")
        )
    )

    return edges_map_df.join(nodes_map_df, on=main_cols + ["home_loc"], how="left")


def get_OD(
    sdf_pairs: DataFrame,
    base_cols: Sequence[str],
    time_agg_cols: Sequence[str],
):
    """
    Compute origin-destination (OD) aggregates from consecutive location pairs.
    This function expects `sdf_pairs` to contain consecutive transitions, including the columns:
    - `loc` (origin), `_next_loc` (destination)
    - `latitude`, `longitude`, `_next_lat`, `_next_lon`
    and it aggregates within `main_cols` (e.g., user/time bucket) and OD pair.

    It computes:
    - `count_trips_OD`: number of transitions for each OD pair
    - `mean_pairs_distance_OD`: mean direct distance of those transitions (rounded to 2 decimals)
    - `distance_OD_bin`: a log10-style binned label: "1-10 km", "10-100 km", etc.
    """

    main_cols = list(base_cols) + list(time_agg_cols)

    sdf_OD = (
        sdf_pairs.withColumn(
            "_dist_direct_edge",
            distance(
                F.col("latitude"),
                F.col("longitude"),
                F.col("_next_lat"),
                F.col("_next_lon"),
            ),
        )
        .withColumnRenamed("loc", "origin_id")
        .withColumnRenamed("_next_loc", "destination_id")
        .groupBy(*main_cols, "origin_id", "destination_id")
        .agg(
            F.count(F.lit(1)).alias("count_trips_OD"),
            F.round(F.mean("_dist_direct_edge"), 2).alias("mean_pairs_distance_OD"),
        )
        # Bin distances into log10 bins: 0-1 km, 1-10 km, 10-100 km, etc.
        .withColumn(
            "distance_bin_idx",
            F.when(F.col("mean_pairs_distance_OD").isNull(), F.lit(None)).otherwise(
                F.floor(F.log10(F.col("mean_pairs_distance_OD"))).cast("int")
            ),
        )
        # clamp anything below 1 km into idx=0
        .withColumn(
            "distance_bin_idx",
            F.when(F.col("distance_bin_idx").isNull(), F.lit(None))
            .when(F.col("distance_bin_idx") < 0, F.lit(0))
            .otherwise(F.col("distance_bin_idx")),
        )
        # build readable log-distance label "1-10 km", "10-100 km", etc.
        .withColumn(
            "distance_bin_OD",
            F.when(F.col("distance_bin_idx").isNull(), F.lit(None)).otherwise(
                F.concat(
                    F.pow(F.lit(10.0), F.col("distance_bin_idx")).cast("int"),
                    F.lit("-"),
                    F.pow(F.lit(10.0), F.col("distance_bin_idx") + F.lit(1)).cast(
                        "int"
                    ),
                    F.lit(" km"),
                )
            ),
        )
        # drop columns
        .drop("distance_bin_idx", "mean_pairs_distance_OD")
    )

    return sdf_OD.orderBy(*main_cols, "origin_id", "destination_id")
