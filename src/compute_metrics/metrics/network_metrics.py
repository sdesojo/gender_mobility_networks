# from pyspark.sql import DataFrame, Window
# from pyspark.sql import functions as F
from typing import Sequence, Optional, Mapping, Dict, Any
import networkx as nx


# -- PREPARE GRAPH DATA ----------------------------------------
def build_graph(
    nodes_weights: Optional[Mapping[str, int]],
    edges_weights: Optional[Mapping[str, int]],
    edge_delim: str = "-",
) -> nx.Graph:
    """Build an undirected NetworkX graph from node/edge weight maps."""
    G = nx.Graph()

    if nodes_weights:
        for node, w in nodes_weights.items():
            # Cast node to str to be safe across spark/pandas typing
            G.add_node(str(node), weight=int(w) if w is not None else 0)

    if edges_weights:
        for edge_key, w in edges_weights.items():
            if edge_key is None:
                continue
            parts = str(edge_key).split(edge_delim)
            if len(parts) != 2:
                continue
            u, v = parts[0], parts[1]
            if u == v:
                continue
            G.add_edge(u, v, weight=int(w) if w is not None else 0)

    return G


def drop_selfloops(G: nx.Graph) -> nx.Graph:
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def largest_connected_component_subgraph(G: nx.Graph) -> nx.Graph:
    """Return the subgraph induced by the largest connected component (or empty graph)"""
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()


# -- DEFINE NETWORK METRICS ----------------------------------------


# ---- simple helpers ----
def _round(x, nd=6):
    return float(round(x, nd)) if x is not None else None


# ---- basic counts ----
def nodes(G: nx.Graph) -> Dict[str, Any]:
    return {"nodes": int(G.number_of_nodes())}


def edges(G: nx.Graph) -> Dict[str, Any]:
    return {"edges": int(G.number_of_edges())}


# ---- connectivity ----
def network_density(G: nx.Graph) -> Dict[str, Any]:
    if G.number_of_nodes() <= 1:
        return {"density": None}
    return {"density": _round(nx.density(G))}


def avg_node_connectivity(G: nx.Graph) -> Dict[str, Any]:
    # Can be expensive for large graphs
    n = G.number_of_nodes()
    if n <= 1:
        return {"avg_node_connectivity": None}

    # Optional safety guard:
    # if n > 600:  # tune for your data
    # return {"avg_node_connectivity": None}

    return {"avg_node_connectivity": _round(nx.average_node_connectivity(G))}


def edge_connectivity(G: nx.Graph) -> Dict[str, Any]:
    n = G.number_of_nodes()
    if n <= 1:
        return {"edge_connectivity": None}

    # Optional safety guard:
    # if n > 1000:
    #     return {"edge_connectivity": None}

    return {"edge_connectivity": int(nx.edge_connectivity(G))}


# ---- clustering / triangles ----
def avg_unw_cluster(G: nx.Graph) -> Dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {"avg_unw_cluster": None}
    return {"avg_unw_cluster": _round(nx.average_clustering(G, weight=None))}


def avg_cluster_weighted(G: nx.Graph) -> Dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {"avg_weighted_cluster": None}
    # Uses 'weight' edge attribute as strength
    return {
        "avg_weighted_cluster": _round(nx.average_clustering(G, weight="weight"), 6)
    }


def transitivity(G: nx.Graph) -> Dict[str, Any]:
    if G.number_of_nodes() <= 2:
        return {"transitivity": None}
    return {"transitivity": _round(nx.transitivity(G), 6)}


# ---- cycles ----
def cycles(G: nx.Graph) -> Dict[str, Any]:
    cycles = nx.cycle_basis(G) if G.number_of_nodes() > 0 else []
    if not cycles:
        return {
            "n_cycles": 0,
            "avg_len_cycles_all": 0.0,
            "n_cycles_per_edge": 0.0,
        }  # n_triangles_per_edge
    cycles = nx.cycle_basis(G)
    lens = [len(c) for c in cycles]
    triangles = len([l for l in lens if l == 3])

    return {
        "n_cycles": int(len(cycles)),
        "avg_len_cycles_all": _round(sum(lens) / len(lens)),
        "n_cycles_per_edge": (
            _round(len(cycles) / G.number_of_edges())
            if G.number_of_edges() > 0
            else None
        ),
        # "n_triangles_per_edge": _round(triangles / G.number_of_edges()) if G.number_of_edges() > 0 else None
    }


# ---- centrality (top-k) ----
def degree_centrality_top(G: nx.Graph) -> Dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {
            "top1_degree_centrality": None,
            "top2_degree_centrality": None,
            "top3_degree_centrality": None,
        }

    dc = nx.degree_centrality(G)
    vals = sorted(dc.values(), reverse=True) if dc else []
    return {
        "top1_degree_centrality": _round(vals[0]) if len(vals) >= 1 else None,
        "top2_degree_centrality": _round(vals[1]) if len(vals) >= 2 else None,
        "top3_degree_centrality": _round(vals[2]) if len(vals) >= 3 else None,
    }


def betweenness_centrality_top(G: nx.Graph) -> Dict[str, Any]:
    """
    Expensive: O(V*E) typically. Keep top3 only.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {
            "top1_betweenness": None,
            "top2_betweenness": None,
            "top3_betweenness": None,
        }

    # Optional safety guard:
    # if n > 2000:
    #     return {"top1_betweenness": None, "top2_betweenness": None, "top3_betweenness": None}

    bc = nx.betweenness_centrality(
        G, weight=None
    )  # weight=None => unweighted shortest paths
    vals = sorted(bc.values(), reverse=True) if bc else []
    return {
        "top1_betweenness": _round(vals[0]) if len(vals) >= 1 else None,
        "top2_betweenness": _round(vals[1]) if len(vals) >= 2 else None,
        "top3_betweenness": _round(vals[2]) if len(vals) >= 3 else None,
    }


def closeness_centrality_top(G: nx.Graph) -> Dict[str, Any]:
    n = G.number_of_nodes()
    if n == 0:
        return {"top1_closeness": None, "top2_closeness": None, "top3_closeness": None}

    # Optional safety guard:
    # if n > 5000:
    #     return {"top1_closeness": None, "top2_closeness": None, "top3_closeness": None}

    cc = nx.closeness_centrality(G)
    vals = sorted(cc.values(), reverse=True) if cc else []
    return {
        "top1_closeness": _round(vals[0], 6) if len(vals) >= 1 else None,
        "top2_closeness": _round(vals[1], 6) if len(vals) >= 2 else None,
        "top3_closeness": _round(vals[2], 6) if len(vals) >= 3 else None,
    }


def eigenvector_centrality_top(G: nx.Graph) -> Dict[str, Any]:
    """
    Can fail to converge. Use try/except and return None on failure.
    Uses edge 'weight' as strength.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {
            "top1_eigenvector": None,
            "top2_eigenvector": None,
            "top3_eigenvector": None,
        }

    try:
        ec = nx.eigenvector_centrality(G, weight="weight", max_iter=500)
    except Exception:
        return {
            "top1_eigenvector": None,
            "top2_eigenvector": None,
            "top3_eigenvector": None,
        }

    vals = sorted(ec.values(), reverse=True) if ec else []
    return {
        "top1_eigenvector": _round(vals[0], 6) if len(vals) >= 1 else None,
        "top2_eigenvector": _round(vals[1], 6) if len(vals) >= 2 else None,
        "top3_eigenvector": _round(vals[2], 6) if len(vals) >= 3 else None,
    }


# ---- distance metrics ----
def eccentricity_stats(G: nx.Graph) -> Dict[str, Any]:
    """
    NetworkX eccentricity returns a dict node->eccentricity. We summarize it as max/mean.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {"eccentricity_max": None, "eccentricity_mean": None}

    # eccentricity requires connected graph; your pipeline uses LCC so should be connected.
    # But guard anyway:
    if n > 1 and not nx.is_connected(G):
        return {"eccentricity_max": None, "eccentricity_mean": None}

    # Optional safety guard (eccentricity all-pairs shortest paths)
    # if n > 2000:
    #     return {"eccentricity_max": None, "eccentricity_mean": None}

    ecc = nx.eccentricity(G)  # dict
    vals = list(ecc.values()) if ecc else []
    if not vals:
        return {"eccentricity_max": None, "eccentricity_mean": None}

    return {
        "eccentricity_max": int(max(vals)),
        "eccentricity_mean": _round(sum(vals) / len(vals)),
    }


def diameter(G: nx.Graph) -> Dict[str, Any]:
    n = G.number_of_nodes()
    if n <= 1:
        return {"diameter": None}

    # diameter requires connected graph; LCC should be connected:
    if not nx.is_connected(G):
        return {"diameter": None}

    # Optional safety guard:
    # if n > 2000:
    #     return {"diameter": None}

    return {"diameter": int(nx.diameter(G))}


# ---- efficiency ----
def local_efficiency(G: nx.Graph) -> Dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {"local_efficiency": None}
    return {"local_efficiency": _round(nx.local_efficiency(G))}


def global_efficiency(G: nx.Graph) -> Dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {"global_efficiency_edges": None}
    return {"global_efficiency_edges": _round(nx.global_efficiency(G))}


# -- DEFINE NETWORK EFFICIENCY METRICS [moved to network_efficiency_metrics.py]----------------------------------------

# def global_efficiency_distance(G: nx.Graph) -> Dict[str, Any]:
#     return None # > implement if we want it; can be expensive (all pairs shortest paths)


# schema_structure = DoubleType()
# @F.udf(schema_structure)
# def get_geo_global_efficency(edges_lij: dict, edges_lij_to_ideal: dict):
#     """
#     Returns the average global efficiency of the graph [1], considering the physical distance between nodes.
#     Given edges weighted with great-circle distance for the orginal graph edges and the edges missing
#     to get the fully conncected graph, get the weighted G (with weight as 'gc_dist') and the weighted Gideal.
#     Then get shortest path lenght weighted with gc-distance for G and Gid, and
#     calculates global efficency considering gc-distances, following fomula (7) in [2].

#     (7) E_glob = [1 / N(N - 1)] * \sum_{i != j} (l_ij/d_ij).

#     Parameters
#     ----------
#     edges_lij : class:`dict`
#         A dict of edges, including the great-circle distance between ndoes as weight
#     edges_lij_to_ideal : class:`dict`
#         A dict of the edges missing to get to the ideal graph, including the great-circle distance between them.

#     Returns
#     -------
#     float
#         The average global efficiency of the graph.

#     Notes
#     -----
#     The *efficiency* of a pair of nodes in a graph is the multiplicative
#     inverse of the shortest path distance between the nodes. The *average
#     global efficiency* of a graph is the average efficiency of all pairs of
#     nodes [1]_.


#     References
#     ----------
#     .. [1] Latora, Vito, and Massimo Marchiori.
#            "Efficient behavior of small-world networks."
#            *Physical Review Letters* 87.19 (2001): 198701.
#            <https://doi.org/10.1103/PhysRevLett.87.198701>

#     .. [2] VRAGOVIC, LOUIS, AND DÍAZ-GUILERA
#            "Efficiency of informational transfer in regular and complex networks."
#            *Physical Review Letters* 71.3 (2005): 036122.
#            <https://doi.org/10.1103/PhysRevE.71.036122>

#     """
#     ## GET REAL NW
#     def get_nw(edges_lij):
#         ## Create Network of visisted stops
#         G = nx.Graph()

#         # Add edges with edge-weights
#         for edge, weight in edges_lij.items():
#             source, target = map(str, edge.split('-'))
#             G.add_edge(source, target, gc_dist=weight)

#         # print(G.edges(data=True))
#         return G

#     ## GET IDEAL NW
#     def get_nw_ideal(G, edges_lij_to_ideal):
#         Gid = G.copy()
#         # Add missing edges with edge-weights to get ideal
#         for edge, weight in edges_lij_to_ideal.items():
#             source, target = map(str, edge.split('-'))
#             Gid.add_edge(source, target, gc_dist=weight)
#         return Gid


#     ## GET GEO GLOBAL EFFICENCY
#     def get_geo_globeff(G, Gid):
#         n = len(G)
#         denom = n * (n - 1)
#         if denom != 0:
#             Dij = dict(nx.shortest_path_length(G, weight = 'gc_dist'))
#             Lij = dict(nx.shortest_path_length(Gid, weight = 'gc_dist'))
#             g_eff = 0
#             for source, targets in Dij.items():
#                 for target, dij in targets.items():
#                     lij = dict(Lij)[source][target]
#                     if dij > 0:
#                         g_eff += lij / dij
#             g_eff /= denom

#         else:
#             g_eff = 0

#         return g_eff

#     ## APPLY
#     G = get_nw(edges_lij)
#     Gid = get_nw_ideal(G, edges_lij_to_ideal)
#     glob_eff = get_geo_globeff(G, Gid)

#     return float(round(glob_eff,8))
