from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List, Dict, Any
from compute_metrics.metrics.registry import METRIC_REGISTRY


class MobilityMetricsPipeline:

    def __init__(
        self, sdf: DataFrame, agg_cfg: Dict[str, Any], met_cfg: Dict[str, Any]
    ):
        self.sdf = sdf
        self.agg_cfg = agg_cfg

        # ---- Metric names from YAML ----
        self.metric_names: List[str] = met_cfg["mobility_metrics"]
        self.aux_metric: List[str] = met_cfg["mobility_aux_metrics"]
        self.metric_funcs = [METRIC_REGISTRY[m] for m in self.metric_names]

        # ---- Aggregation config ----
        self.base_cols = agg_cfg["base_cols"]
        self.sort_cols = agg_cfg["time_sort_cols"]
        self.time_agg_cols = agg_cfg["time_agg_cols"]
        self.suffix = agg_cfg.get("suffix", "")

    def _apply_metrics(self) -> DataFrame:
        sdf = self.sdf

        for func in self.metric_funcs:
            sdf = func(
                sdf,
                base_cols=self.base_cols,
                time_sort_cols=self.sort_cols,
                time_agg_cols=self.time_agg_cols,
                suffix=self.suffix,
            )

        # keep final sdf and keep selected auxiliary mobility metrics
        self.sdf = sdf
        return sdf

    def _aggregate(self) -> DataFrame:
        """Aggregate all metric columns once per base_cols + temporal resolution."""

        sdf = self.sdf
        group_cols = self.base_cols + self.time_agg_cols
        metric_cols = [f"{name}{self.suffix}" for name in self.metric_names] + [
            f"{name}{self.suffix}" for name in self.aux_metric
        ]

        agg_exprs = [
            F.max(c).alias(c) for c in metric_cols
        ]  # THEY ARE ALREADY COMPUTED AT MONTH LEVEL

        sdf_agg = sdf.groupBy(*group_cols).agg(*agg_exprs)

        return sdf_agg

    def run(self) -> DataFrame:
        self._apply_metrics()
        return self._aggregate()
