from compute_metrics.metrics.mobility_metrics import *
from compute_metrics.metrics.network_metrics import *
from pyspark.sql import types as T


METRIC_REGISTRY = {
    # --- Mobility metrics ---
    "visits": visits,
    "locations": locations,
    "total_stay_time": total_stay_time,
    "total_travel_time": total_travel_time,
    "total_cost_travel_distance": total_cost_travel_distance,
    "total_reward_home_distance": total_reward_home_distance,
    "tour_efficiency_dist": tour_efficiency_dist,
    # --- Network metrics ---
    "nodes": nodes,
    "edges": edges,
    "density": network_density,
    "avg_node_connectivity": avg_node_connectivity,
    "edge_connectivity": edge_connectivity,
    "local_efficiency": local_efficiency,
    "global_efficiency_edges": global_efficiency,
    # These return multiple keys; if you keep them, YAML + schema must include all keys!
    "eccentricity_stats": eccentricity_stats,
    "diameter": diameter,
    "transitivity": transitivity,
    "avg_weighted_cluster": avg_cluster_weighted,
    "avg_unw_cluster": avg_unw_cluster,
    "top1_degree_centrality": degree_centrality_top,
    "n_cycles": cycles,
    "betweenness_centrality_top": betweenness_centrality_top,
    "closeness_centrality_top": closeness_centrality_top,
    "eigenvector_centrality_top": eigenvector_centrality_top,
}

METRIC_SCHEMA = {
    "nodes": T.IntegerType(),
    "edges": T.IntegerType(),
    "density": T.DoubleType(),
    "avg_unw_cluster": T.DoubleType(),
    "transitivity": T.DoubleType(),
    "avg_weighted_cluster": T.DoubleType(),
    "avg_node_connectivity": T.DoubleType(),
    "edge_connectivity": T.IntegerType(),
    # cycles outputs:
    "n_cycles": T.IntegerType(),
    "avg_len_cycles_all": T.DoubleType(),
    "n_cycles_per_edge": T.DoubleType(),
    # eccentricity_stats outputs:
    "eccentricity_max": T.IntegerType(),
    "eccentricity_mean": T.DoubleType(),
    "diameter": T.IntegerType(),
    # degree_centrality_top outputs:
    "top1_degree_centrality": T.DoubleType(),
    "top2_degree_centrality": T.DoubleType(),
    "top3_degree_centrality": T.DoubleType(),
    # betweenness top:
    "top1_betweenness": T.DoubleType(),
    "top2_betweenness": T.DoubleType(),
    "top3_betweenness": T.DoubleType(),
    # closeness top:
    "top1_closeness": T.DoubleType(),
    "top2_closeness": T.DoubleType(),
    "top3_closeness": T.DoubleType(),
    # eigenvector top:
    "top1_eigenvector": T.DoubleType(),
    "top2_eigenvector": T.DoubleType(),
    "top3_eigenvector": T.DoubleType(),
    "local_efficiency": T.DoubleType(),
    "global_efficiency_edges": T.DoubleType(),
    # Efficiency:
    # "global_efficiency_distance": T.DoubleType(),
    # "mfpt": T.DoubleType(),
}
