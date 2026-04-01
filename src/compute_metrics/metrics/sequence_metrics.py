# from __future__ import annotations
# import re
# from math import acos, cos, sin, radians
# from typing import Any, Dict, List, Sequence, Set, Tuple

# import numpy as np
# from pyspark.sql import DataFrame
# import pyspark.sql.functions as F
# from pyspark.sql.types import (
#     StructType, StructField, ArrayType, StringType, LongType, DoubleType, IntegerType
# )
# from pyspark.sql.functions import udf

# # ---------- distance ----------
# def distance(lat_p1: float, lon_p1: float, lat_p2: float, lon_p2: float) -> float:
#     if (lat_p1 == lat_p2) and (lon_p1 == lon_p2):
#         return 0.0
#     return acos(
#         sin(radians(lat_p1)) * sin(radians(lat_p2)) +
#         cos(radians(lat_p1)) * cos(radians(lat_p2)) *
#         cos(radians(lon_p1) - radians(lon_p2))
#     ) * 6371.0

# # ---------- sequences ----------
# def get_daily_stop_sequences(df: DataFrame, TEMP_RESO: Sequence[str]) -> DataFrame:
#     df = df.withColumn("lat_lon", F.struct("lat", "lon"))
#     df_struct = df.withColumn("stop_struct", F.struct("start", "end", "stops", 'lat_lon'))
#     grouped = (
#         df_struct
#         .groupBy(["useruuid"] + list(TEMP_RESO) + ['EmpStatus_Mu'])
#         .agg(F.sort_array(F.collect_list("stop_struct")).alias("ordered_stops"))
#     )
#     return (
#         grouped
#         .withColumn("stop_sequence", F.expr("transform(ordered_stops, x -> x.stops)"))
#         .withColumn("arr_time",     F.expr("transform(ordered_stops, x -> x.start)"))
#         .withColumn("dep_time",     F.expr("transform(ordered_stops, x -> x.end)"))
#         .withColumn("positions",    F.expr("transform(ordered_stops, x -> x.lat_lon)"))
#         .withColumn("seq_len", F.size("stop_sequence"))
#         .select("useruuid", *TEMP_RESO, 'EmpStatus_Mu',
#                 "stop_sequence", "arr_time", "dep_time", "positions", "seq_len")
#     )

# # ---------- clean consecutive stops UDF ----------
# position_struct = StructType([StructField("lat", DoubleType()), StructField("lon", DoubleType())])
# seq_schema = StructType([
#     StructField("stop_sequence", ArrayType(StringType())),
#     StructField("stop_time", ArrayType(LongType())),
#     StructField("arr_time", ArrayType(LongType())),
#     StructField("dep_time", ArrayType(LongType())),
#     StructField("positions", ArrayType(position_struct)),
#     StructField("seq_len", LongType()),
# ])

# @udf(seq_schema)
# def udf_clean_seq_consec_stops(sequence, positions, arr_time, dep_time):
#     if not sequence or not positions or not arr_time or not dep_time:
#         return {"stop_sequence": [], "stop_time": [], "arr_time": [], "dep_time": [],
#                 "positions": [], "seq_len": 0}

#     clean_seq, clean_coords, clean_arr, clean_dep = [sequence[0]], [positions[0]], [arr_time[0]], [dep_time[0]]
#     for i in range(1, len(sequence)):
#         if sequence[i] != clean_seq[-1]:
#             clean_seq.append(sequence[i]); clean_coords.append(positions[i])
#             clean_arr.append(arr_time[i]); clean_dep.append(dep_time[i])
#         else:
#             clean_dep[-1] = dep_time[i] # if its the same stop keep the last departure time
#     clean_stop_time = [d - a for a, d in zip(clean_arr, clean_dep)]
#     return {"stop_sequence": clean_seq, "stop_time": clean_stop_time,
#             "arr_time": clean_arr, "dep_time": clean_dep,
#             "positions": clean_coords, "seq_len": len(clean_seq)}

# # ---------- journeys/keystones ----------
# journey_schema = StructType([
#     StructField("start_idx", LongType()),
#     StructField("end_idx", LongType()),
#     StructField("places", ArrayType(StringType())),
#     StructField("distances", ArrayType(LongType())),
#     StructField("distances_from_start", ArrayType(LongType())),
#     StructField("stay_departure", LongType()),
#     StructField("total_duration", LongType()),
# ])
# jk_schema = StructType([
#     StructField("journeys", ArrayType(journey_schema)),
#     StructField("filtered_journeys", ArrayType(journey_schema)),
#     StructField("keystones", ArrayType(StringType())),
# ])

# def _find_journeys(sequence, arrival_times, departure_times, positions):
#     journeys, stack = [], []
#     for idx, loc in enumerate(sequence):
#         matches = [i for i, (L, _) in enumerate(stack) if L == loc]
#         if matches:
#             dpos = matches[0]
#             start_idx = stack[dpos][1]
#             places = sequence[start_idx: idx+1]
#             coords_ = positions[start_idx: idx+1]
#             try: coords = [(float(c.lat), float(c.lon)) for c in coords_ if c is not None]
#             except: coords = [(float(c['lat']), float(c['lon'])) for c in coords_ if c is not None] # helper for examples

#             stay_departure = departure_times[start_idx] - arrival_times[start_idx]
#             total_duration = arrival_times[idx] - departure_times[start_idx]
#             distances = [distance(*coords[i], *coords[i+1]) for i in range(len(coords)-1)]
#             lat0, lon0 = coords[0]
#             distances_from_start = [distance(lat0, lon0, lat, lon) for lat, lon in coords]
#             journeys.append({
#                 'start_idx': start_idx, 'end_idx': idx, 'places': places,
#                 'distances': [round(1000*d) for d in distances],
#                 'distances_from_start': [round(1000*d) for d in distances_from_start],
#                 'stay_departure': stay_departure, 'total_duration': total_duration
#             })

#             # Remove everything from the stack after that departure index and treat as a fresh departure
#             stack = stack[:dpos]
#             stack.append((loc, idx))
#         else:
#             # No matching departure: mark this location as a new open departure
#             stack.append((loc, idx))
#     return journeys

# def _find_keystones(journeys) -> Set[str]:
#     ks = set()
#     for j in journeys:
#         if j['total_duration'] < j['stay_departure']:
#             ks.add(j['places'][0])
#     return ks

# def _filter_journeys_from_keystones(journeys, keystones):
#     J = [j for j in journeys if j['places'][0] in keystones]
#     return [j for j in J if len(set(j['places'])) > 1] # is keystone and has more than one location

# @udf(jk_schema)
# def udf_extract_journeys_and_keystones(sequence, arr_times, dep_times, positions):
#     if not sequence or not arr_times or not dep_times:
#         return [], [], []
#     journeys = _find_journeys(sequence, arr_times, dep_times, positions)
#     keystones = list(_find_keystones(journeys))
#     filtered = _filter_journeys_from_keystones(journeys, keystones)
#     return journeys, filtered, keystones

# @udf(StringType())
# def udf_get_main_keystone(keystones, stop_sequence, arr_time, dep_time):
#     if not keystones or len(keystones) < 1:
#         return None
#     k_tottime = {}
#     for k in keystones:
#         idxs = [i for i, v in enumerate(stop_sequence) if v == k]
#         times = [dep_time[int(i)] - arr_time[int(i)] for i in idxs]
#         k_tottime[k] = sum(times)
#     return max(k_tottime, key=k_tottime.get) if k_tottime else None

# # ---------- sequence distances/efficiency ----------
# dist_schema = StructType([
#     StructField("seq_distances", ArrayType(LongType())),
#     StructField("seq_dist_from_start", ArrayType(LongType())),
#     StructField("seq_dist_from_home", ArrayType(LongType())),
#     StructField("seq_dist_from_keystone", ArrayType(LongType())),
#     StructField("seq_efficency_start", DoubleType()),
#     StructField("seq_savings_start", DoubleType()),
#     StructField("seq_efficency_home", DoubleType()),
#     StructField("seq_savings_home", DoubleType()),
#     StructField("seq_efficency_keystone", DoubleType()),
#     StructField("seq_savings_keystone", DoubleType()),
# ])

# @udf(dist_schema)
# def udf_get_seq_distances(sequence, positions, main_keystone):
#     if len(sequence) < 2:
#         return {k: ([] if 'seq_' in k else None) for k in dist_schema.names}
#     if any(p is None or p.lat is None or p.lon is None for p in positions):
#         return {k: ([] if 'seq_' in k else None) for k in dist_schema.names}

#     coords = [(float(p.lat), float(p.lon)) for p in positions]
#     distances = [distance(*coords[i], *coords[i+1]) for i in range(len(coords)-1)]
#     lat0, lon0 = coords[0]
#     dist_from_start = [distance(lat0, lon0, lat, lon) for lat, lon in coords]

#     # home
#     home_idxs = [i for i, loc in enumerate(sequence) if loc == 'H']
#     dist_from_home = []
#     if home_idxs:
#         h0 = coords[home_idxs[0]]
#         dist_from_home = [distance(h0[0], h0[1], lat, lon) for lat, lon in coords]

#     # keystone
#     ks_idxs = [i for i, loc in enumerate(sequence) if loc == main_keystone]
#     dist_from_keystone = []
#     if ks_idxs:
#         k0 = coords[ks_idxs[0]]
#         dist_from_keystone = [distance(k0[0], k0[1], lat, lon) for lat, lon in coords]

#     cost = sum(distances)
#     reward_start = 2 * sum(dist_from_start)
#     reward_home = 2 * sum(dist_from_home)
#     reward_keystone = 2 * sum(dist_from_keystone)

#     eff = lambda cost, reward: (1 - cost / reward) if reward > 0 else None
#     sav = lambda cost, reward: (reward - cost) if reward > 0 else None

#     return {
#         "seq_distances": [round(d) for d in distances],
#         "seq_dist_from_start": [round(d) for d in dist_from_start],
#         "seq_dist_from_home": [round(d) for d in dist_from_home],
#         "seq_dist_from_keystone": [round(d) for d in dist_from_keystone],
#         "seq_log_"
#         "seq_efficency_start": round(eff(cost, reward_start), 4) if eff(cost, reward_start) is not None else None,
#         "seq_savings_start": round(sav(cost, reward_start), 4) if sav(cost, reward_start) is not None else None,
#         "seq_efficency_home": round(eff(cost, reward_home), 4) if eff(cost, reward_home) is not None else None,
#         "seq_savings_home": round(sav(cost, reward_home), 4) if sav(cost, reward_home) is not None else None,
#         "seq_efficency_keystone": round(eff(cost, reward_keystone), 4) if eff(cost, reward_keystone) is not None else None,
#         "seq_savings_keystone": round(sav(cost, reward_keystone), 4) if sav(cost, reward_keystone) is not None else None,
#     }

# # ---------- journey summary ----------
# summary_schema = StructType([
#     StructField("jrny_total", IntegerType()),
#     StructField("jrny_total_unique", IntegerType()),
#     StructField("jrny_lists", ArrayType(ArrayType(StringType()))),
#     StructField("jrny_avg_length", DoubleType()),
#     StructField("jrny_med_length", LongType()),
#     StructField("jrny_avg_length_unique", DoubleType()),
#     StructField("jrny_med_length_unique", LongType()),
#     StructField("jrny_avg_duration", DoubleType()),
#     StructField("jrny_sum_dist", ArrayType(LongType())),
#     StructField("jrny_sum_dist_from_start", ArrayType(LongType())),
# ])

# @udf(summary_schema)
# def udf_summarize_journeys(journeys):
#     if not journeys:
#         return {"jrny_total": 0, "jrny_total_unique": 0, "jrny_lists": [],
#                 "jrny_avg_length": 0.0, "jrny_med_length": 0, "jrny_avg_length_unique": 0.0,
#                 "jrny_med_length_unique": 0, "jrny_avg_duration": 0.0,
#                 "jrny_sum_dist": [], "jrny_sum_dist_from_start": []}

#     places_lists = [j['places'] for j in journeys if 'places' in j]
#     lengths = [len(p) for p in places_lists]
#     unique_places = list({tuple(p) for p in places_lists})
#     lengths_unique = [len(p) for p in unique_places]
#     durations = [j['total_duration'] for j in journeys if 'total_duration' in j]
#     distances_j = [j['distances'] for j in journeys if 'distances' in j]
#     distances_from_start_j = [j['distances_from_start'] for j in journeys if 'distances_from_start' in j]

#     import numpy as np
#     return {
#         "jrny_total": len(places_lists),
#         "jrny_total_unique": len(unique_places),
#         "jrny_lists": places_lists,
#         "jrny_avg_length": float(np.mean(lengths)) if lengths else 0.0,
#         "jrny_med_length": int(np.median(lengths)) if lengths else 0,
#         "jrny_avg_length_unique": float(np.mean(lengths_unique)) if lengths_unique else 0.0,
#         "jrny_med_length_unique": int(np.median(lengths_unique)) if lengths_unique else 0,
#         "jrny_avg_duration": float(np.mean(durations)) if durations else 0.0,
#         "jrny_sum_dist": [sum(d) for d in distances_j] if distances_j else [],
#         "jrny_sum_dist_from_start": [sum(d) for d in distances_from_start_j] if distances_from_start_j else [],
#     }

# # ---------- reward columns (sequence & journey) ----------
# def add_sequence_reward_columns(sdf: DataFrame) -> DataFrame:
#     return (
#         sdf
#         .withColumn('seq_tot_cost', F.aggregate("seq_distances", F.lit(0.0), lambda acc, x: acc + x))
#         .withColumn('seq_tot_reward_home', 2 * F.aggregate("seq_dist_from_home", F.lit(0.0), lambda acc, x: acc + x))
#         .withColumn('seq_tot_reward_keystone', 2 * F.aggregate("seq_dist_from_keystone", F.lit(0.0), lambda acc, x: acc + x))
#         .withColumn('seq_max_reward_home', 2 * F.array_max("seq_dist_from_home"))
#         .withColumn('seq_max_reward_keystone', 2 * F.array_max("seq_dist_from_keystone"))
#         .withColumn("seq_clean_home", F.expr("filter(seq_dist_from_home, x -> x != 0)"))
#         .withColumn("seq_clean_keystone", F.expr("filter(seq_dist_from_keystone, x -> x != 0)"))
#         .withColumn("seq_mean_reward_home",
#             F.when(F.size("seq_clean_home") > 0,
#                    2 * (F.aggregate("seq_clean_home", F.lit(0.0), lambda acc, x: acc + x) / F.size("seq_clean_home")))
#         )
#         .withColumn("seq_mean_reward_keystone",
#             F.when(F.size("seq_clean_keystone") > 0,
#                    2 * (F.aggregate("seq_clean_keystone", F.lit(0.0), lambda acc, x: acc + x) / F.size("seq_clean_keystone")))
#         )
#         .withColumn("seq_median_reward_home",
#             F.when(F.size("seq_clean_home") > 0,
#                    2 * F.element_at(F.sort_array("seq_clean_home"),
#                                     ((F.size("seq_clean_home") + F.lit(1)) / 2).cast("int")))
#         )
#         .withColumn("seq_median_reward_keystone",
#             F.when(F.size("seq_clean_keystone") > 0,
#                    2 * F.element_at(F.sort_array("seq_clean_keystone"),
#                                     ((F.size("seq_clean_keystone") + F.lit(1)) / 2).cast("int")))
#         )
#     )

# def add_jrny_reward_columns(sdf: DataFrame) -> DataFrame:
#     sdf_seq_j = sdf.withColumn("journeys", F.expr("""
#         transform(journeys, j -> struct(
#             j.start_idx as start_idx,
#             j.end_idx as end_idx,
#             j.places as places,
#             j.distances as distances,
#             j.distances_from_start as distances_from_start,
#             j.stay_departure as stay_departure,
#             j.total_duration as total_duration,
#             size(j.places) as jrny_lenght,
#             aggregate(j.distances, 0L, (acc, x) -> acc + x) as jrny_tot_cost,
#             2 * aggregate(j.distances_from_start, 0L, (acc, x) -> acc + x) as jrny_tot_reward,
#             2 * array_max(j.distances_from_start) as jrny_max_reward,
#             slice(j.distances_from_start, 2, size(j.distances_from_start) - 2) as jrny_clean_start
#         ))
#     """))
#     return sdf_seq_j.withColumn("journeys", F.expr("""
#         transform(journeys, j -> struct(
#             j.start_idx as start_idx,
#             j.end_idx as end_idx,
#             j.places as places,
#             j.distances as distances,
#             j.distances_from_start as distances_from_start,
#             j.stay_departure as stay_departure,
#             j.total_duration as total_duration,
#             j.jrny_lenght as jrny_lenght,
#             j.jrny_tot_cost as jrny_tot_cost,
#             j.jrny_tot_reward as jrny_tot_reward,
#             j.jrny_max_reward as jrny_max_reward,
#             2 * (aggregate(j.jrny_clean_start, 0D, (acc, x) -> acc + x) / size(j.jrny_clean_start)) as jrny_mean_reward,
#             2 * element_at(sort_array(j.jrny_clean_start), cast((size(j.jrny_clean_start) + 1) / 2 as int)) as jrny_median_reward
#         ))
#     """))
