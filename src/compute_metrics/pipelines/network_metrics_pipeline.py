from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

import warnings

warnings.simplefilter("ignore", FutureWarning)

from typing import List, Dict, Any, Sequence

import pandas as pd

from compute_metrics.metrics.registry import METRIC_REGISTRY, METRIC_SCHEMA
from compute_metrics.metrics.network_builder import (
    get_nodes_edges,
    compress_nodes_edges_to_maps,
    get_OD,
)
from compute_metrics.metrics.network_metrics import (
    build_graph,
    drop_selfloops,
    largest_connected_component_subgraph,
)
from compute_metrics.metrics.network_efficiency_metrics import (
    compute_mfpt_metrics,
    get_edges_distances,
    get_edges_distances_ideal,
    compute_global_eff_dist,
)


class NetworkMetricsPipeline:

    def __init__(
        self,
        sdf: DataFrame,
        agg_cfg: Dict[str, Any],
        met_cfg: Dict[str, Any],
        out_netw_dir: str,
    ):
        self.sdf = sdf
        self.agg_cfg = agg_cfg
        self.out_netw_dir = out_netw_dir

        # ---- Metric names from YAML ----
        self.nw_metric_names: List[str] = met_cfg["network_metrics"]
        self.aux_metric: List[str] = met_cfg["network_aux_metrics"]
        self.eff_metric_names: List[str] = met_cfg["efficiency_metrics"]

        # ---- Aggregation config ----
        self.base_cols = agg_cfg["base_cols"]
        self.sort_cols = agg_cfg["time_sort_cols"]
        self.time_agg_cols = agg_cfg["time_agg_cols"]
        self.suffix = agg_cfg.get("suffix")
        self.get_OD_netw = agg_cfg["get_OD_netw"]

        # ---- Graph build options (forced defaults) ----
        self.edge_delim: str = "-"
        self.use_lcc: bool = True
        self.loc_col = "loc"

        # ---- Metric output schema (this defines self.metric_schema.fields) ----
        missing = [m for m in self.nw_metric_names if m not in METRIC_REGISTRY]
        if missing:
            raise KeyError(f"Missing registry for metrics: {missing}")

        all_metrics = list(self.nw_metric_names) + list(self.aux_metric)
        missing = [m for m in all_metrics if m not in METRIC_SCHEMA]
        if missing:
            raise KeyError(f"Missing schema for metrics: {missing}")
        # self.metric_schema = T.StructType([
        #     T.StructField(f"{m}{self.suffix}", METRIC_SCHEMA[m], True) for m in all_metrics
        # ])

        self.metric_schema = T.StructType(
            [
                T.StructField(f"{m}{self.suffix}", METRIC_SCHEMA[m], True)
                for m in METRIC_SCHEMA.keys()
                if m in all_metrics
            ]
        )

    def _build_network_maps(self) -> DataFrame:
        sdf = self.sdf

        # Get weighted nodes, edges for undirected networks
        sdf_nodes, sdf_edges, sdf_pairs = get_nodes_edges(
            sdf,
            base_cols=self.base_cols,
            time_sort_cols=self.sort_cols,
            time_agg_cols=self.time_agg_cols,
            loc_col=self.loc_col,
            suffix=self.suffix,
        )
        # print("> Built nodes and edges DataFrames")

        # Optional: save the OD pairs
        if self.get_OD_netw:
            sdf_OD = get_OD(sdf_pairs, self.base_cols, self.time_agg_cols)

            sdf_OD.write.mode("overwrite").parquet(self.out_netw_dir)
            print("> Saved OD pairs DataFrame")

        # Compress to per user-month maps using pure Spark
        main_cols = list(self.base_cols) + list(self.time_agg_cols)

        sdf_maps = compress_nodes_edges_to_maps(
            sdf_nodes=sdf_nodes,
            sdf_edges=sdf_edges,
            base_cols=self.base_cols,
            time_sort_cols=self.sort_cols,
            time_agg_cols=self.time_agg_cols,
            loc_col=self.loc_col,
            node_w_col=f"node_weight_und{self.suffix}",
            edge_pair_col=f"edges_pairs_undirected{self.suffix}",
            edge_w_col=f"edge_weight_und{self.suffix}",
            edge_delim=self.edge_delim,
        )

        return sdf_maps, sdf_pairs

    def _apply_nw_metrics(self, sdf_maps: DataFrame) -> DataFrame:

        id_cols = list(self.base_cols) + list(self.time_agg_cols)
        nw_metric_names = list(self.nw_metric_names)  # + list(self.aux_metric)
        metric_funcs = [METRIC_REGISTRY[m] for m in nw_metric_names]
        # print(f"> Applying {len(metric_funcs)} network metrics: {nw_metric_names}")

        # Output schema: id columns + metric columns (already suffixed in self.metric_schema)
        out_schema = T.StructType(
            [T.StructField(c, sdf_maps.schema[c].dataType, True) for c in id_cols]
            + list(self.metric_schema.fields)
        )

        edge_delim = self.edge_delim  # forced "-"
        use_lcc = self.use_lcc  # forced True

        # These are the output column names (possibly suffixed)
        out_metric_cols = [f.name for f in self.metric_schema.fields]
        # print(f"> Output metric columns: {out_metric_cols}")

        # Map output column -> base metric name (remove suffix if present)
        # Example: "density_und" -> "density"
        if self.suffix:
            base_metric_names = [
                c[: -len(self.suffix)] if c.endswith(self.suffix) else c
                for c in out_metric_cols
            ]
        else:
            base_metric_names = out_metric_cols

        def _iter_metrics(pdf_iter):
            for pdf in pdf_iter:
                out_rows = []

                for _, r in pdf.iterrows():
                    G = build_graph(
                        r.get("nodes_weights") or {},
                        r.get("edges_weights") or {},
                        edge_delim=edge_delim,
                    )

                    # Get largest component and drop self-loops
                    if use_lcc:
                        G_eval = largest_connected_component_subgraph(drop_selfloops(G))
                    else:
                        G_eval = G

                    d = {}
                    for fn in metric_funcs:
                        # metrics return dict like {"density": ...}
                        d.update(fn(G_eval))

                    row_out = {c: r[c] for c in id_cols}

                    # Fill output columns in stable order, applying suffix
                    for out_col, base_name in zip(out_metric_cols, base_metric_names):
                        row_out[out_col] = d.get(base_name, None)

                    out_rows.append(row_out)

                yield pd.DataFrame(out_rows)

        sdf_in = sdf_maps.select(*id_cols, "nodes_weights", "edges_weights")
        return sdf_in.mapInPandas(_iter_metrics, schema=out_schema)

    def _apply_efficiency_metrics(
        self, sdf_maps: DataFrame, sdf_pairs: DataFrame
    ) -> DataFrame:

        main_cols = list(self.base_cols) + list(self.time_agg_cols)

        # Start from one base DF (one row per group)
        sdf_eff = sdf_maps.select(*main_cols, "home_loc", "edges_weights")

        # --- Global efficiency distance (likely from sdf_pairs) ---
        if "global_efficiency_distance" in self.eff_metric_names:
            sdf_edge_dist = get_edges_distances(sdf_pairs, main_cols)
            sdf_edge_dist_ideal = get_edges_distances_ideal(sdf_pairs, main_cols)
            sdf_ge = compute_global_eff_dist(
                sdf_edge_dist, sdf_edge_dist_ideal, main_cols
            )  # returns main_cols + global_efficiency_distance
            sdf_eff = sdf_eff.join(sdf_ge, on=main_cols, how="left")

        # --- MFPT (from sdf_maps) ---
        if (
            "nwide_MFPT_lovasz" in self.eff_metric_names
            or "home_GMFPT_lovasz" in self.eff_metric_names
        ):
            sdf_eff = (
                sdf_eff
                # .where(F.col("home_loc") != "missing_home") # applied only for home_GMFPT, but not for nwide_MFPT, since the latter can be computed even without a home detected
                .withColumn(
                    "_mfpt",
                    compute_mfpt_metrics(F.col("edges_weights"), F.col("home_loc")),
                )
                .withColumn("nwide_MFPT_lovasz", F.col("_mfpt")["nwide_MFPT"])
                .withColumn("home_GMFPT_lovasz", F.col("_mfpt")["home_GMFPT"])
                .drop("_mfpt")
            )

        # print("> Applied efficiency metrics")

        # Final projection: keep only what you promised
        out_cols = list(
            dict.fromkeys(main_cols + list(self.eff_metric_names))
        )  # stable + de-dupe
        return sdf_eff.select(*out_cols)

    def run(self) -> DataFrame:
        sdf_maps, sdf_pairs = self._build_network_maps()
        sdf_metrics = self._apply_nw_metrics(sdf_maps)
        sdf_eff = self._apply_efficiency_metrics(sdf_maps, sdf_pairs)

        sdf_out = sdf_metrics.join(
            sdf_eff, on=list(self.base_cols) + list(self.time_agg_cols), how="left"
        )
        return sdf_out
