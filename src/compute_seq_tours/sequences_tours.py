from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from typing import Sequence, List, Dict, Any, Set, Union
from pyspark.sql.types import (
    DoubleType,
    ArrayType,
    StringType,
    StructType,
    StructField,
    IntegerType,
    LongType,
)
from pyspark.sql.functions import udf
import math
import pandas as pd

from utils.utils import distance_py


## 1. Prepare stop sequences per user and temporal resolution
def get_stop_sequences(
    sdf: DataFrame, base_cols: Sequence[str], time_agg_cols: Sequence[str]
) -> DataFrame:
    """
    Transforms a DataFrame of individual stops into ordered stop sequences grouped
    by user and temporal resolution.

    For each group, stops are collected into arrays sorted chronologically by start
    time, and their locations, arrival times, departure times, and coordinates are
    extracted as parallel arrays.

    Args:
        sdf: Input DataFrame with columns for stop location (loc, latitude, longitude),
             timing (start, end), and all columns in base_cols and time_agg_cols.
        base_cols: Base grouping columns (e.g. ["useruuid"]).
        time_agg_cols: Temporal resolution columns to group by (e.g. ["date"] or
                       ["year", "month"]).

    Returns:
        DataFrame grouped by (base_cols + time_agg_cols) with columns:
            - stop_sequence: ordered array of stop location identifiers (loc)
            - arr_time:      ordered array of arrival timestamps (start)
            - dep_time:      ordered array of departure timestamps (end)
            - positions:     ordered array of (latitude, longitude) structs
            - seq_len:       number of stops in the sequence
    """
    # Create struct for sorting and preservation
    sdf = sdf.withColumn("lat_lon", F.struct("latitude", "longitude"))
    sdf_struct = sdf.withColumn(
        "stop_struct", F.struct("start", "end", "loc", "lat_lon")
    )

    # Group by user and temporal resolution
    main_cols = base_cols + time_agg_cols
    grouped = sdf_struct.groupBy(main_cols).agg(
        F.sort_array(F.collect_list("stop_struct")).alias("ordered_stops")
        # Sorts the collected list of stop_struct by the 'start' timestamp, ensuring the sequence is in chronological order.
    )

    # Extract components
    result_df = (
        grouped.withColumn(
            "stop_sequence", F.expr("transform(ordered_stops, x -> x.loc)")
        )
        .withColumn("arr_time", F.expr("transform(ordered_stops, x -> x.start)"))
        .withColumn("dep_time", F.expr("transform(ordered_stops, x -> x.end)"))
        .withColumn("positions", F.expr("transform(ordered_stops, x -> x.lat_lon)"))
        .withColumn("seq_len", F.size("stop_sequence"))
        .select(
            *main_cols,
            "stop_sequence",
            "arr_time",
            "dep_time",
            "positions",
            "seq_len",
        )
    )

    return result_df


## 2. Drop self-loops, update stop times and positions accordingly
# Define schema for dists sequences
position_struct = StructType(
    [StructField("lat", DoubleType()), StructField("lon", DoubleType())]
)

seq_schema = StructType(
    [
        StructField("stop_sequence", ArrayType(StringType())),
        StructField("stop_time", ArrayType(LongType())),
        StructField("arr_time", ArrayType(LongType())),
        StructField("dep_time", ArrayType(LongType())),
        StructField("positions", ArrayType(position_struct)),
        StructField("seq_len", LongType()),
    ]
)


@udf(seq_schema)
def udf_clean_seq_consec_stops(sequence, positions, arr_time, dep_time):
    if not sequence or not positions or not arr_time or not dep_time:
        return {
            "stop_sequence": [],
            "stop_time": [],
            "arr_time": [],
            "dep_time": [],
            "positions": [],
            "seq_len": 0,
        }

    clean_seq = [sequence[0]]
    clean_coords = [positions[0]]
    clean_arr_time = [arr_time[0]]
    clean_dep_time = [dep_time[0]]

    for i in range(1, len(sequence)):
        if sequence[i] != clean_seq[-1]:
            clean_seq.append(sequence[i])
            clean_coords.append(positions[i])

            clean_arr_time.append(arr_time[i])
            clean_dep_time.append(dep_time[i])

        else:  # if its the same stop keep
            clean_dep_time[-1] = dep_time[i]  # Update only the last dep time

    clean_stop_time = [d - a for a, d in zip(clean_arr_time, clean_dep_time)]
    return {
        "stop_sequence": clean_seq,
        "stop_time": clean_stop_time,
        "arr_time": clean_arr_time,
        "dep_time": clean_dep_time,
        "positions": clean_coords,
        "seq_len": len(clean_seq),
    }


# 3. Identify tours from stop sequences
def find_tours_with_times_distances(
    sequence: list,
    arrival_times: list,
    departure_times: list,
    positions: list,
) -> list[dict]:
    """
    Identifies tours within a stop sequence using a stack-based matching algorithm.

    A tour is defined as a departure from a location and a subsequent return to that
    same location. When a repeated location is detected, the earliest unmatched
    departure is closed and recorded as a tour. Nested tours (e.g. A→B→A→C→A) are
    handled correctly — the inner tour is closed first, and the outer location is
    re-pushed as a fresh departure.

    Args:
        sequence:        Ordered list of visited locations.
        arrival_times:   Arrival timestamp at each stop, parallel to sequence.
        departure_times: Departure timestamp at each stop, parallel to sequence.
        positions:       Coordinates at each stop as objects or dicts with lat/lon
                         fields, parallel to sequence.

    Returns:
        List of tours, where each tour is a dict with:
            start_idx:            Index in sequence where the tour departs.
            end_idx:              Index in sequence where the tour returns.
            places:               Ordered list of locations from start to end (inclusive).
            distances:            Distances in metres between consecutive stops.
            distances_from_start: Distances in metres from the departure location
                                  to each stop.
            stay_departure:       Time spent at the departure location before leaving
                                  (departure_time - arrival_time at start_idx).
            total_duration:       Total travel time from departure to return
                                  (arrival_time at end_idx - departure_time at start_idx).
    """

    tours = []
    stack = []  # Holds tuples of (location, index) for open departures

    for idx, loc in enumerate(sequence):
        # Find any open departure matching this location
        matches = [i for i, (stack_loc, _) in enumerate(stack) if stack_loc == loc]

        if matches:
            # Close the earliest matching departure
            dpos = matches[0]
            start_idx = stack[dpos][1]

            # Extract the journey’s “places”, coordinates and timing
            places = sequence[start_idx : idx + 1]

            coords_ = positions[start_idx : idx + 1]
            try:
                coords = [
                    (float(c.lat), float(c.lon)) for c in coords_ if c is not None
                ]
            except:
                coords = [
                    (float(c["lat"]), float(c["lon"])) for c in coords_ if c is not None
                ]  # only for examples

            stay_departure = departure_times[start_idx] - arrival_times[start_idx]
            total_duration = arrival_times[idx] - departure_times[start_idx]

            # Distances between consecutive locations (km)
            distances = [
                distance_py(
                    coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]
                )
                for i in range(len(coords) - 1)
            ]

            # Distances to start location (km)
            lat0, lon0 = coords[0]
            distances_from_start = [
                distance_py(lat0, lon0, lat, lon) for lat, lon in coords
            ]

            tours.append(
                {
                    "start_idx": start_idx,
                    "end_idx": idx,
                    "places": places,
                    "distances": [round(1000 * d) for d in distances],
                    "distances_from_start": [
                        round(1000 * d) for d in distances_from_start
                    ],
                    "stay_departure": stay_departure,
                    "total_duration": total_duration,
                }
            )

            # Remove everything from the stack after that departure index
            stack = stack[:dpos]
            # Now treat this arrival as a fresh departure
            stack.append((loc, idx))
        else:
            # No matching departure: mark this location as a new open departure
            stack.append((loc, idx))

    return tours


## 4. Identify keystone locations from tours
def find_keystones(tours: List[Dict[str, Any]]) -> Set[str]:
    """
    Identify keystone locations from a list of tours.
    A tour is considered a keystone-originating trip if: trip duration is smaller than time spent at departure, i.e.
      total_duration < stay_departure.

    Parameters:
        tours: List of dicts, each with keys:
            - 'places': list of locations (departure is places[0])
            - 'stay_departure': time spent at departure
            - 'total_duration': trip duration from departure to return

    Returns:
        A set of locations (strings) that are keystones.
    """
    keystones = set()
    for j in tours:
        if j["total_duration"] < j["stay_departure"]:
            # The departure location is the first in 'places'
            keystones.add(j["places"][0])
    return keystones


## 5. Filter tours, to keep only tours from keystones. Keep at least 2 different locations (to avoid loops on the same location)
def filter_tours_from_keystones(tours, keystones):
    """
    Return only the tours:
        - whose departure location is in keystones,
        - that have at least 2 different locations,
    """
    # Filter by starting location
    filter_tours = [j for j in tours if j["places"][0] in keystones]

    # Filter by having at least two different locations
    filter_tours = [j for j in filter_tours if len(set(j["places"])) > 1]

    return filter_tours


## 6. UDF to implement steps 3-5: extract tours and keystones from stop sequences
# Define schema for tours
journey_schema = StructType(
    [
        StructField("start_idx", LongType()),
        StructField("end_idx", LongType()),
        StructField("places", ArrayType(StringType())),
        StructField("distances", ArrayType(LongType())),  # meters
        StructField("distances_from_start", ArrayType(LongType())),  # meters
        StructField("stay_departure", LongType()),  # seconds
        StructField("total_duration", LongType()),  # seconds
    ]
)

# UDF return schema
schema = StructType(
    [
        StructField("tours", ArrayType(journey_schema)),
        StructField("filtered_tours", ArrayType(journey_schema)),
        StructField("keystones", ArrayType(StringType())),
    ]
)


# Register UDF
@udf(schema)
def udf_extract_tours_and_keystones(sequence, arr_times, dep_times, positions):

    if not sequence or not arr_times or not dep_times:
        return [], []

    # All inputs are in Unix time already
    tours = find_tours_with_times_distances(sequence, arr_times, dep_times, positions)
    keystones = list(find_keystones(tours))
    filtered_tours = filter_tours_from_keystones(tours, keystones)

    return tours, filtered_tours, keystones


# 7. Keep main keystone per sequence (the one with the most total time spent on tours departing from it)
@udf(StringType())
def udf_get_main_keystone(keystones, stop_sequence, arr_time, dep_time):
    if len(keystones) < 1:
        return None

    k_tottime = {}
    for k in keystones:
        k_idx = [i for i, val in enumerate(stop_sequence) if val == k]
        k_time = [dep_time[int(i)] - arr_time[int(i)] for i in k_idx]
        k_tottime[k] = sum(k_time)

    # Get the keystone with the maximum total time
    max_k = max(k_tottime, key=k_tottime.get)
    return max_k


# 8. [Validation purposes only] Get travel efficiency across different anchors: keystone, home, first stop.
# Define schema for dists sequences
dist_schema = StructType(
    [
        StructField("seq_distances", ArrayType(LongType())),  # meters
        StructField("seq_cost", DoubleType()),  # meters
        StructField("seq_dist_from_start", ArrayType(LongType())),  # meters
        StructField("seq_dist_from_home", ArrayType(LongType())),  # meters
        StructField("seq_dist_from_keystone", ArrayType(LongType())),  # meters
        StructField("seq_reward_start", DoubleType()),  # meters
        StructField("seq_efficency_start", DoubleType()),
        StructField("seq_savings_start", DoubleType()),
        StructField("seq_reward_home", DoubleType()),  # meters
        StructField("seq_efficency_home", DoubleType()),
        StructField("seq_savings_home", DoubleType()),
        StructField("seq_reward_keystone", DoubleType()),  # meters
        StructField("seq_efficency_keystone", DoubleType()),
        StructField("seq_savings_keystone", DoubleType()),
    ]
)


@udf(dist_schema)
def udf_get_seq_distances(sequence, positions, main_keystone):

    # Skip if sequence contains only one place
    empty_return = {
        "seq_distances": [],
        "seq_cost": None,
        "seq_dist_from_start": [],
        "seq_dist_from_home": [],
        "seq_dist_from_keystone": [],
        "seq_reward_start": None,
        "seq_efficency_start": None,
        "seq_savings_start": None,
        "seq_reward_home": None,
        "seq_efficency_home": None,
        "seq_savings_home": None,
        "seq_reward_keystone": None,
        "seq_efficency_keystone": None,
        "seq_savings_keystone": None,
    }

    if len(sequence) < 2:
        return empty_return

    # Skip if any postions element is None
    if any(p is None or p.lat is None or p.lon is None for p in positions):
        return empty_return

    # Extract coordinates as floats
    coords = [(float(p.lat), float(p.lon)) for p in positions]

    # Distances between consecutive locations (meters)
    distances = [
        1000
        * distance_py(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        for i in range(len(coords) - 1)
    ]

    # Distances to start location (meters)
    lat0, lon0 = coords[0]
    distances_from_start = [
        1000 * distance_py(lat0, lon0, lat, lon) for lat, lon in coords
    ]

    # Distances to home location (meters)
    k_idxs = [i for i, loc in enumerate(sequence) if loc == "H"]
    distances_from_home = []
    if k_idxs:
        lat0, lon0 = coords[k_idxs[0]]
        distances_from_home = [
            1000 * distance_py(lat0, lon0, lat, lon) for lat, lon in coords
        ]

    # Distances to main keystone (meters)
    k_idxs = [i for i, loc in enumerate(sequence) if loc == main_keystone]
    distances_from_keystone = []

    if k_idxs:
        lat0, lon0 = coords[k_idxs[0]]
        distances_from_keystone = [
            1000 * distance_py(lat0, lon0, lat, lon) for lat, lon in coords
        ]

    # Efficiency: (r-c)/c -> savings/reward:
    #  “How much travel did I avoid by chaining visits instead of going back and forth from a central place (the keystone)?”
    # eff = 0: just enough movement to reach destination. Your route is as long as (or nearly as long as) returning to the keystone each time.
    # eff = 1: perfect efficiency. You saved almost all the travel you would have done if you returned to the keystone each time.
    # Apply log to avoid etxreme values biasing the results
    # Distances are already stored in meters

    cost = sum(distances)

    reward_start = 2 * sum(distances_from_start)
    reward_home = 2 * sum(distances_from_home)
    reward_keystone = 2 * sum(distances_from_keystone)

    eff_seq_start = (
        1 - math.log10(cost) / math.log10(reward_start) if reward_start > 0 else None
    )
    eff_savings_start = reward_start - cost if (reward_start > 0) else None

    eff_seq_home = (
        1 - math.log10(cost) / math.log10(reward_home)
        if (reward_home > 0 and k_idxs)
        else None
    )
    eff_savings_home = reward_home - cost if (reward_home > 0 and k_idxs) else None

    eff_seq_keystone = (
        1 - math.log10(cost) / math.log10(reward_keystone)
        if (reward_keystone > 0 and k_idxs)
        else None
    )
    eff_savings_keystone = (
        reward_keystone - cost if (reward_keystone > 0 and k_idxs) else None
    )

    return {
        "seq_distances": [round(d, 4) for d in distances],
        "seq_cost": round(cost, 4) if cost is not None else None,
        "seq_dist_from_start": [round(d, 4) for d in distances_from_start],
        "seq_dist_from_keystone": [round(d, 4) for d in distances_from_keystone],
        "seq_dist_from_home": [round(d, 4) for d in distances_from_home],
        "seq_reward_start": (
            round(reward_start, 4) if reward_start is not None else None
        ),
        "seq_efficency_start": (
            round(eff_seq_start, 4) if eff_seq_start is not None else None
        ),
        "seq_savings_start": (
            round(eff_savings_start, 4) if eff_savings_start is not None else None
        ),
        "seq_reward_home": round(reward_home, 4) if reward_home is not None else None,
        "seq_efficency_home": (
            round(eff_seq_home, 4) if eff_seq_home is not None else None
        ),
        "seq_savings_home": (
            round(eff_savings_home, 4) if eff_savings_home is not None else None
        ),
        "seq_reward_keystone": (
            round(reward_keystone, 4) if reward_keystone is not None else None
        ),
        "seq_efficency_keystone": (
            round(eff_seq_keystone, 4) if eff_seq_keystone is not None else None
        ),
        "seq_savings_keystone": (
            round(eff_savings_keystone, 4) if eff_savings_keystone is not None else None
        ),
    }


# 9. Get Journey metrics: lenght, distance, duration.
# -- Enrich each tour struct with computed metrics (pure Spark, no UDF)
def add_jrny_reward_columns(sdf):
    return sdf.withColumn(
        "tours",
        F.expr(
            """
            transform(filtered_tours, j -> struct(
                j.places                                                        as places,
                size(j.places)                                                  as jrny_length,
                aggregate(j.distances, 0L, (acc, x) -> acc + x)                as jrny_cost,
                2 * aggregate(j.distances_from_start, 0L, (acc, x) -> acc + x) as jrny_reward,
                2 * array_max(j.distances_from_start)                           as jrny_max_reward,
                j.total_duration                                                as jrny_duration
            ))
        """
        ),
    )


summary_schema = StructType(
    [
        StructField("jrny_total", IntegerType()),
        StructField("jrny_total_unique", IntegerType()),
        StructField("jrny_lists", ArrayType(ArrayType(StringType()))),
        StructField("jrny_length", ArrayType(IntegerType())),
        StructField("jrny_cost", ArrayType(LongType())),
        StructField("jrny_reward", ArrayType(LongType())),
        StructField("jrny_max_reward", ArrayType(LongType())),
        StructField("jrny_duration", ArrayType(LongType())),  # seconds
    ]
)


@udf(summary_schema)
def udf_summarize_tours(tours):

    empty_return = {
        "jrny_total": 0,
        "jrny_total_unique": 0,
        "jrny_lists": [],
        "jrny_length": [],
        "jrny_cost": [],
        "jrny_reward": [],
        "jrny_max_reward": [],
        "jrny_duration": [],
    }

    if not tours:
        return empty_return

    places_lists = [j["places"] for j in tours]
    lengths = [j["jrny_length"] for j in tours]
    costs = [j["jrny_cost"] for j in tours]
    rewards = [j["jrny_reward"] for j in tours]
    max_rewards = [j["jrny_max_reward"] for j in tours]
    durations = [j["jrny_duration"] for j in tours]

    unique_places = list({tuple(p) for p in places_lists})

    return {
        "jrny_total": len(places_lists),
        "jrny_total_unique": len(unique_places),
        "jrny_lists": places_lists,
        "jrny_length": lengths,
        "jrny_cost": costs,
        "jrny_reward": rewards,
        "jrny_max_reward": max_rewards,
        "jrny_duration": durations,
    }


####
# 99. Get length bins for journeys, to be used in the analysis and visualizations.
def get_length_bin(df_jrny, leng_met="jrny_length", minv=3, maxv=5):
    MAXv = max(df_jrny[leng_met])

    fixed_bins = [i + 0.5 for i in range(minv - 1, maxv + 1)] + [
        6.5,
        10.5,
        20.5,
        float("inf"),
    ]
    fixed_labels = ["3\nABA", "4\nABCA", "5\nABCDA", "6", "7-10", "11-20", "20+"]
    # print(fixed_bins)
    # print(fixed_labels)

    df_jrny = df_jrny[df_jrny[leng_met] >= minv].copy()
    df_jrny["leng_bin"] = pd.cut(
        df_jrny[leng_met], bins=fixed_bins, labels=fixed_labels, include_lowest=True
    )
    return df_jrny
