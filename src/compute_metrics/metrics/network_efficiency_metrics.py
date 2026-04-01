from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import networkx as nx

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
)

import pandas as pd
from utils.utils import distance


# -- GET GLOBAL EFFICIENCY [not networkX implementation] ----------------------------------------
def get_edges_distances(sdf_pairs: DataFrame, main_cols: List[str]) -> DataFrame:
    """
    Given a DataFrame of mobility edges (with columns "_und_edge", "_next_lat", "_next_lon"),
    compute the great-circle distance for each edge.
    """
    sdf_edge_dist = (
        sdf_pairs.withColumn(
            "_dist_direct_edge",
            distance(
                F.col("latitude"),
                F.col("longitude"),
                F.col("_next_lat"),
                F.col("_next_lon"),
            ),
        )
        .groupBy(*main_cols, "_und_edge")
        .agg(
            F.count(F.lit(1)).alias("count_trips_OD"),
            F.round(F.mean("_dist_direct_edge"), 2).alias("mean_pairs_distance_OD"),
        )
        .withColumn("ei", F.col("_und_edge")[0])
        .withColumn("ej", F.col("_und_edge")[1])
        .drop("_und_edge")
        .select(*main_cols, "ei", "ej", "count_trips_OD", "mean_pairs_distance_OD")
    )

    return sdf_edge_dist.orderBy(*main_cols, "ei", "ej")


def get_edges_distances_ideal(sdf_pairs: DataFrame, main_cols: List[str]) -> DataFrame:
    """
    Ideal version of get_edges_distances, where we compute the distance for all possible edges between observed nodes,
    rather than just the observed edges.
    """
    # 1) Get unique nodes with their lat/lon
    sdf_nodes = (
        sdf_pairs.select(
            *main_cols,
            F.explode(
                F.array(
                    F.struct(
                        F.col("loc").alias("node"),
                        F.col("latitude").alias("node_lat"),
                        F.col("longitude").alias("node_lon"),
                    ),
                    F.struct(
                        F.col("_next_loc").alias("node"),
                        F.col("_next_lat").alias("node_lat"),
                        F.col("_next_lon").alias("node_lon"),
                    ),
                )
            ).alias("n"),
        )
        .select(*main_cols, "n.node", "n.node_lat", "n.node_lon")
        .dropDuplicates(main_cols + ["node"])
    )

    # 2) Get distance between all pairs of nodes (not self-loops)
    n1 = sdf_nodes.alias("n1")
    n2 = sdf_nodes.alias("n2")
    sdf_edge_dist = (
        n1.join(n2, on=main_cols, how="inner")
        .where(F.col("n1.node") < F.col("n2.node"))
        .select(
            *[F.col(f"n1.{c}").alias(f"{c}") for c in main_cols],
            F.col("n1.node").alias("ei"),
            F.col("n1.node_lat").alias("ei_lat"),
            F.col("n1.node_lon").alias("ei_lon"),
            F.col("n2.node").alias("ej"),
            F.col("n2.node_lat").alias("ej_lat"),
            F.col("n2.node_lon").alias("ej_lon"),
        )
        .withColumn(
            "_dist_und_edge",
            distance(
                F.col("ei_lat"), F.col("ei_lon"), F.col("ej_lat"), F.col("ej_lon")
            ),
        )
        .withColumn("mean_pairs_distance_ideal", F.round(F.col("_dist_und_edge"), 2))
        .drop("ei_lat", "ei_lon", "ej_lat", "ej_lon", "_dist_und_edge")
    )

    return sdf_edge_dist


def compute_global_eff_dist(
    sdf_edge_dist: DataFrame, sdf_edge_dist_ideal: DataFrame, main_cols: List[str]
) -> DataFrame:
    """
    Compute global efficiency based on shortest path lengths in the mobility network.
    This is a distance-based version of global efficiency, where we treat edge weights as distances.

    Args:
        sdf_pairs: DataFrame with columns main_cols + ["_und_edge", "_next_lat", "_next_lon"] representing the mobility graph edges.
        main_cols: List of columns that define the grouping (e.g., user_id, month).

    Returns:
        DataFrame with columns main_cols + ["global_efficiency_distance"] containing the computed metric.
    """
    # 1. Build a graph for each group defined by main_cols using the edges in sdf_pairs and log distance.
    # 2. Compute shortest path lengths between all pairs of nodes in the OBSERVED AND IDEAL graphs.
    # 3. Calculate global efficiency as the average of 1 / shortest_path_length for all pairs of nodes.

    def _get_geo_globeff(G, Gid):
        n = len(G)
        denom = n * (n - 1)
        if denom != 0:
            Dij = dict(nx.shortest_path_length(G, weight="gc_dist"))
            Lij = dict(nx.shortest_path_length(Gid, weight="gc_dist"))
            g_eff = 0
            for source, targets in Dij.items():
                for target, dij in targets.items():

                    lij = dict(Lij)[source][target]
                    if dij > 0:
                        g_eff += lij / dij
            g_eff /= denom

        else:
            g_eff = 0

        return g_eff

    def _run_per_group(pdf: pd.DataFrame) -> pd.DataFrame:
        keys = {c: pdf[c].iloc[0] for c in main_cols}

        # --- Build Gid (ideal complete graph for observed nodes) ---
        Gid = nx.Graph()
        for ei, ej, l in zip(pdf["ei"], pdf["ej"], pdf["l_ij"]):
            if pd.notna(l):
                Gid.add_edge(str(ei), str(ej), gc_dist=float(l))

        # --- Build G (observed graph) ---
        G = nx.Graph()
        obs = pdf.dropna(subset=["d_ij"])
        for ei, ej, d in zip(obs["ei"], obs["ej"], obs["d_ij"]):
            G.add_edge(str(ei), str(ej), gc_dist=float(d))

        # # Make sure node sets align (important for your denom and for Lij lookups)
        # G.add_nodes_from(Gid.nodes()) # -> already match given construction (see bellow, join by ei, ej)

        ge = float(round(_get_geo_globeff(G, Gid), 8))
        return pd.DataFrame([{**keys, "global_efficiency_distance": ge}])

    # Join so each group has both observed and ideal edge distances
    sdf = (
        sdf_edge_dist_ideal
        # ensure same undirected edge order is applied
        .withColumn(
            "und_edge",
            F.sort_array(
                F.array(F.col("ei").cast("int"), F.col("ej").cast("int")), asc=True
            ),
        )
        .withColumn("ei", F.col("und_edge")[0])
        .withColumn("ej", F.col("und_edge")[1])
        # Get log10 distance (in meters) to avoid extreme values
        .withColumn(
            "l_ij",
            F.when(
                F.col("mean_pairs_distance_ideal") > 0,
                F.log10(F.lit(1000) * F.col("mean_pairs_distance_ideal")),
            ).otherwise(F.lit(None)),
        )
        .select(*main_cols, "ei", "ej", "l_ij")
        .join(
            sdf_edge_dist
            # ensure same undirected edge order is applied
            .withColumn(
                "und_edge",
                F.sort_array(
                    F.array(F.col("ei").cast("int"), F.col("ej").cast("int")), asc=True
                ),
            )
            .withColumn("ei", F.col("und_edge")[0])
            .withColumn("ej", F.col("und_edge")[1])
            # Get log10 distance (in meters) to avoid extreme values
            .withColumn(
                "d_ij",
                F.when(
                    F.col("mean_pairs_distance_OD") > 0,
                    F.log10(F.lit(1000) * F.col("mean_pairs_distance_OD")),
                ).otherwise(F.lit(None)),
            ).select(*main_cols, "ei", "ej", "d_ij"),
            on=main_cols + ["ei", "ej"],
            how="left",
        )
    )

    out_schema = StructType(
        [StructField(c, sdf.schema[c].dataType, True) for c in main_cols]
        + [StructField("global_efficiency_distance", DoubleType(), True)]
    )

    return sdf.groupBy(*main_cols).applyInPandas(_run_per_group, schema=out_schema)


# -- GET RANDOM DIFFUSION METRICS ----------------------------------------


def get_adjacency_mx(edges_weights: Dict[str, float], home_node: str) -> np.ndarray:
    """
    Build an undirected weighted adjacency matrix from an edge-weight dict, placing `home_node` at index 0.

    Args:
        edges_weights: Dict mapping "u-v" -> weight (edge delimiter must be "-").
                      Node ids are expected to be parseable as ints.
        home_node: Node id (string or int-like) to place at index 0 in the adjacency matrix.

    Returns:
        A: (N, N) numpy array adjacency matrix with node order [home_node] + sorted(other_nodes).

    Raises:
        ValueError: If `home_node` is not present in the graph.
    """
    G = nx.Graph()
    G.add_weighted_edges_from(
        (int(e.split("-")[0]), int(e.split("-")[1]), float(w))
        for e, w in (edges_weights or {}).items()
    )

    nodes = sorted(G.nodes())

    if (home_node is not None) and (home_node != "missing_home"):
        home = int(home_node)
        if home not in nodes:
            raise ValueError(f"home_node={home} not in graph nodes.")

        nodes.remove(home)
        nodes = [home] + nodes

    return nx.to_numpy_array(G, nodelist=nodes, weight="weight")


def hitting_time_matrix_lovasz(adj):
    """
    Compute the all-pairs hitting time matrix H for a simple,
    undirected, connected graph using Lovasz (1993) spectral formula.

    Parameters
    ----------
    adj : (N, N) array_like
        Adjacency matrix A of the graph. Must be symmetric.

    Returns
    -------
    H : (N, N) ndarray
        Matrix of hitting times H[u, v] (expected steps of a simple
        random walk starting at u to hit v).
    """
    A = np.asarray(adj, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Adjacency matrix must be symmetric (undirected graph).")

    n = A.shape[0]

    # Degrees and edge count
    k = A.sum(axis=1)
    if np.any(k == 0):
        raise ValueError("Graph contains isolated nodes; hitting times are infinite.")
    m = A.sum() / 2.0  # number of edges |E|

    # Normalized adjacency N = D^{-1/2} A D^{-1/2}
    inv_sqrt_deg = 1.0 / np.sqrt(k)
    Dm12 = np.diag(inv_sqrt_deg)
    N = Dm12 @ A @ Dm12

    # Spectral decomposition of N (symmetric -> eigh)
    # eigh returns eigenvalues in ascending order; we reverse to get λ1 ~ 1 first
    lam, W = np.linalg.eigh(N)
    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    W = W[:, idx]

    # α_n = 1 / (1 - λ_n), but α_1 = 0 (skip the trivial eigenvalue λ1 = 1)
    alpha = np.zeros_like(lam)
    alpha[1:] = 1.0 / (1.0 - lam[1:])

    # --- Vectorized Lovasz formula (clear version) ---

    # Recall the formula:
    #   H[u, v] = 2|E| * Σ_{n=2..N} α[n] * ( w[v,n]^2 / k[v]  -  w[u,n] w[v,n] / √(k[u] k[v]) )
    #
    # where α[n] = 1 / (1 - λ[n]) and w[* , n] is eigenvector n.

    # -------------------------------------------------------------
    # 1. Build term:  w[u,n] / sqrt(k[u])
    # -------------------------------------------------------------
    # This will be used for the cross-term w[u,n] w[v,n] / √(k[u] k[v]).

    B = W / np.sqrt(k[:, None])
    # B[u, n] = w[u, n] / sqrt(k[u])

    # -------------------------------------------------------------
    # 2. Build α[n] * (w[u,n] / sqrt(k[u]))
    # -------------------------------------------------------------
    # Multiply each eigenvector column n by α[n].
    # Broadcasting applies α[n] to column n.

    B_alpha = B * alpha[None, :]
    # B_alpha[u, n] = α[n] * w[u,n] / sqrt(k[u])

    # -------------------------------------------------------------
    # 3. Compute the cross term:
    #       S[u,v] = Σ_n α[n] * w[u,n] w[v,n] / √(k[u] k[v])
    # -------------------------------------------------------------
    # This is simply a matrix product:
    #       S = B_alpha @ B.T

    S = B_alpha @ B.T

    # -------------------------------------------------------------
    # 4. Build the diagonal term:
    #       T[v] = Σ_n α[n] * (w[v,n]^2 / k[v])
    # -------------------------------------------------------------
    # First compute w[v,n]^2 / k[v]

    C = (W**2) / k[:, None]
    # C[v, n] = w[v,n]^2 / k[v]

    # Now sum over n with weights α[n] → gives vector length N

    T = C @ alpha
    # T[v] = Σ_n α[n] * (w[v,n]^2 / k[v])

    # -------------------------------------------------------------
    # 5. Combine terms to produce full hitting time matrix
    # -------------------------------------------------------------
    # Broadcast T over rows and subtract S elementwise.

    H = 2 * m * (T[None, :] - S)
    # H[u, v] = 2m * ( T[v] - S[u,v] )

    # -------------------------------------------------------------
    # 6. Numerical cleanup
    # -------------------------------------------------------------
    np.fill_diagonal(H, 0.0)  # H[v,v] = 0
    H[H < 0] = 0.0  # clamp tiny negatives

    return H


def mean_mfpt_over_pairs(H: np.ndarray) -> float:
    """
    MFPT network-wide: Mean MFPT over ordered pairs i != j (all nodes)

    Args:
        H: (N, N) hitting time matrix.

    Returns:
        Mean value of H[i, j] over all i != j.
    """
    n = H.shape[0]
    return float(H.sum() / (n * (n - 1)))


def global_mfpt_node(H: np.ndarray, node_idx: int) -> float:
    """
    Mean MFPT to a target node, averaging over all other sources.

    Args:
        H: (N, N) hitting time matrix.
        node_idx: Column index of the target node.

    Returns:
        Mean MFPT from all i != node_idx to node_idx.
    """
    n = H.shape[0]
    col = H[:, node_idx]  # exclude self (MFPT from node to itself is 0 by construction)
    return float((col.sum() - col[node_idx]) / (n - 1))


# Wrapper to apply in Spark UDF context
_mfpt_schema = StructType(
    [
        StructField("nwide_MFPT", DoubleType(), nullable=True),
        StructField("home_GMFPT", DoubleType(), nullable=True),
    ]
)


@F.udf(_mfpt_schema)
def compute_mfpt_metrics(
    edges_weights: Dict[str, float], home_node: str
) -> Tuple[float, float]:
    """
    Compute MFPT summaries from an undirected weighted graph encoded as an edges-weight dict.

    Args:
        edges_weights: Dict mapping "u-v" -> weight.
        home_node: Node id to place first (index 0) when building adjacency.

    Returns:
        (nwide_MFPT, home_GMFPT):
            nwide_MFPT: mean MFPT over all ordered pairs.
            home_GMFPT: mean MFPT to `home_node` from all other nodes.
        If computation fails (e.g., empty graph), returns (None, None).
    """
    try:
        A = get_adjacency_mx(edges_weights, home_node)
        if A.shape[0] < 2:
            return (None, None)

        H = hitting_time_matrix_lovasz(A)
        mean_GMFPT = mean_mfpt_over_pairs(H)

        if (home_node is not None) and (home_node != "missing_home"):
            home_GMFPT = global_mfpt_node(H, 0)  # home node is at index 0
        else:
            home_GMFPT = None

        return (
            float(round(mean_GMFPT, 8)),
            float(round(home_GMFPT, 8)) if home_GMFPT is not None else None,
        )

    except Exception:
        return (None, None)
