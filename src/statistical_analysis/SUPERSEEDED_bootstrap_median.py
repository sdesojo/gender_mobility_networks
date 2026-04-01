# """
# DIRECTLY FORM GPT!! NEEDS TO BE UDPATED/REVIEWD

# Bootstrap medians by gender and country, with optional population weighting.

# Config YAML (example):
#   data_dir_nw: /work/user/sdsc/gender_dif_entropy/nw_met
#   out_dir: /work/user/sdsc/gender_dif_entropy/bs_med_bydemo
#   tests: ["ie30","mid3080","ae80","all"]         # which dataset suffixes to process
#   use_logged_variant: true                       # uses *_logge.csv if true, else *.csv
#   run_modes: ["mobility_only","across_aeiemid"]  # pick any subset
#   genders: ["MALE","FEMALE"]
#   n_boot: 1000
#   population_weights_csv: null                   # or path to CSV with columns: GID_0, weight

#   # mobility_only (case A)
#   mob_only_vars: ["EntUnc_M","nStops_M","nuStops_M"]

#   # across_aeiemid (case C)
#   control_vars: ["nStops_M","nuStops_M"]
#   main_vars:
#     - EntUnc_M
#     - density
#     - avg_node_connectivity
#     - diameter
#     - avg_cluster
#     - avg_unw_cluster
#     - n_cycles
#     - nu_keystones
#     - avg_len_cycles_all
#     - top1_degree_centrality
#     - top2_degree_centrality
#     - top3_degree_centrality
#     - transitivity
#     - mean_lb_GMFPT
#     - home_lb_GMFPT
#     - global_efficency
#     - log_geo_global_efficency
#     - nodes
#     - edges
# """

# # from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import numpy as np
# import pandas as pd
# # import yaml


# # --------------------------- helpers ---------------------------

# # def load_config(path: str | Path) -> dict:
# #     with open(path, "r") as f:
# #         return yaml.safe_load(f)

# def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
#     total = float(sum(weights.values()))
#     if total == 0:
#         return {k: 0.0 for k in weights}
#     return {k: float(v) / total for k, v in weights.items()}

# def load_population_weights(csv_path: Optional[str | Path]) -> Optional[Dict[str, float]]:
#     if not csv_path:
#         return None
#     dfw = pd.read_csv(csv_path)
#     if "GID_0" not in dfw.columns:
#         raise ValueError("population_weights_csv must have a 'GID_0' column.")
#     # Accept either 'weight' or 'population' column
#     weight_col = "weight" if "weight" in dfw.columns else "population"
#     if weight_col not in dfw.columns:
#         raise ValueError("population_weights_csv needs a 'weight' or 'population' column.")
#     w = dict(zip(dfw["GID_0"].astype(str), dfw[weight_col].astype(float)))
#     return _normalize_weights(w)

# def get_bs_median_se(values: np.ndarray, n_samples: int, rng: np.random.Generator) -> Tuple[float, float]:
#     """Bootstrap the median: return (mean_of_bootstrap_medians, std_of_bootstrap_medians)."""
#     if len(values) == 0:
#         return float("nan"), float("nan")
#     # Draw bootstrap samples of indices for speed
#     idx = rng.integers(0, len(values), size=(n_samples, len(values)), endpoint=False)
#     samples = values[idx]
#     meds = np.median(samples, axis=1)
#     return float(np.mean(meds)), float(np.std(meds, ddof=0))

# def agg_mean_se_results(
#     per_country_means: Sequence[float],
#     per_country_ses: Sequence[float],
#     countries: Sequence[str],
#     pop_weights: Optional[Mapping[str, float]],
#     use_pop_weight: bool,
# ) -> Tuple[float, float]:
#     """
#     Aggregate country-level (mean, se) into global mean and se.
#     If use_pop_weight, weight by normalized pop_weights[GID_0]; otherwise equal weight.
#     """
#     if len(per_country_means) == 0:
#         return float("nan"), float("nan")

#     if use_pop_weight:
#         if not pop_weights:
#             raise ValueError("Population weighting requested but no population weights provided.")
#         w = np.array([float(pop_weights.get(ctry, 0.0)) for ctry in countries], dtype=float)
#     else:
#         w = np.ones(len(countries), dtype=float) / float(len(countries))

#     # Guard against zero-sum weights
#     if w.sum() == 0:
#         w = np.ones(len(countries), dtype=float) / float(len(countries))

#     mean_global = float(np.sum(w * np.array(per_country_means, dtype=float)))
#     # SE aggregation: sqrt of weighted sum of variances (assuming independence)
#     var_global = float(np.sum((w ** 2) * (np.array(per_country_ses, dtype=float) ** 2)))
#     se_global = float(np.sqrt(var_global))
#     return mean_global, se_global


# # --------------------------- core runners ---------------------------

# def _bootstrap_by_gender_country(
#     df: pd.DataFrame,
#     metrics: Sequence[str],
#     genders: Sequence[str],
#     n_boot: int,
#     pop_weights: Optional[Mapping[str, float]],
# ) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
#     """
#     Returns nested dict:
#       result[gender][metric][country] = { 'med': <float>, 'se': <float> }
#       plus 'agg_equalweight' and 'agg_popweight' under each [gender][metric]
#     """
#     out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {g: {m: {} for m in metrics} for g in genders}
#     rng = np.random.default_rng()

#     countries = sorted(df["GID_0"].astype(str).unique().tolist())

#     for gender in genders:
#         dfg = df[df["gender"] == gender]
#         for metric in metrics:
#             per_ctry_mean: List[float] = []
#             per_ctry_se: List[float] = []
#             used_ctry: List[str] = []

#             for ctry in countries:
#                 block = dfg[dfg["GID_0"].astype(str) == ctry]
#                 vals = block[metric].dropna().to_numpy()
#                 med, se = get_bs_median_se(vals, n_boot, rng)
#                 out[gender][metric][ctry] = {"med": med, "se": se}
#                 if np.isfinite(med) and np.isfinite(se):
#                     per_ctry_mean.append(med)
#                     per_ctry_se.append(se)
#                     used_ctry.append(ctry)

#             # Aggregations
#             mean_eq, se_eq = agg_mean_se_results(per_ctry_mean, per_ctry_se, used_ctry, pop_weights, use_pop_weight=False)
#             out[gender][metric]["agg_equalweight"] = {"med": mean_eq, "se": se_eq}

#             mean_pw, se_pw = agg_mean_se_results(per_ctry_mean, per_ctry_se, used_ctry, pop_weights, use_pop_weight=True)
#             out[gender][metric]["agg_popweight"] = {"med": mean_pw, "se": se_pw}

#     return out


# def _run_mobility_only(cfg: dict) -> Path:
#     data_dir = Path(cfg["data_dir_nw"])
#     out_dir = Path(cfg["out_dir"])
#     out_dir.mkdir(parents=True, exist_ok=True)

#     genders = cfg.get("genders", ["MALE", "FEMALE"])
#     n_boot = int(cfg.get("n_boot", 1000))
#     use_logged = bool(cfg.get("use_logged_variant", False))
#     tests = cfg.get("tests", ["all"])
#     metrics = cfg["mob_only_vars"]

#     pop_weights = load_population_weights(cfg.get("population_weights_csv"))

#     for test in tests:
#         fname = f"nwnavmob_mon_{test}{'_logge' if use_logged else ''}.csv"
#         df = pd.read_csv(data_dir / fname)
#         cols = ["useruuid", "year", "month", "gender", "GID_0"] + metrics
#         df = df[cols].copy()

#         result = _bootstrap_by_gender_country(df, metrics, genders, n_boot, pop_weights)
#         out_path = out_dir / f"gender_C_{test}_mob.json"
#         with open(out_path, "w") as f:
#             json.dump(result, f)
#     return out_dir


# def _run_across_aeiemid(cfg: dict) -> Path:
#     data_dir = Path(cfg["data_dir_nw"])
#     out_dir = Path(cfg["out_dir"])
#     out_dir.mkdir(parents=True, exist_ok=True)

#     genders = cfg.get("genders", ["MALE", "FEMALE"])
#     n_boot = int(cfg.get("n_boot", 1000))
#     use_logged = bool(cfg.get("use_logged_variant", False))
#     tests = cfg.get("tests", ["all"])
#     control_vars = cfg["control_vars"]
#     main_vars = cfg["main_vars"]
#     metrics = control_vars + main_vars

#     pop_weights = load_population_weights(cfg.get("population_weights_csv"))

#     for test in tests:
#         fname = f"nwnavmob_mon_{test}{'_logge' if use_logged else ''}.csv"
#         df = pd.read_csv(data_dir / fname)
#         cols = ["useruuid", "year", "month", "gender", "GID_0"] + metrics
#         df = df[cols].dropna().copy()

#         result = _bootstrap_by_gender_country(df, metrics, genders, n_boot, pop_weights)
#         out_path = out_dir / f"gender_C_{test}_mobnwlogge.json" if use_logged else out_dir / f"gender_C_{test}_mobnw.json"
#         with open(out_path, "w") as f:
#             json.dump(result, f)
#     return out_dir


# # --------------------------- public entrypoint ---------------------------

# def run(config_path: str | Path) -> Path:
#     """
#     Entry point called by CLI. Decides which modes to run based on config.
#     """
#     cfg = load_config(config_path)
#     modes = set(cfg.get("run_modes", []))
#     last_out = None

#     if "mobility_only" in modes:
#         last_out = _run_mobility_only(cfg)
#     if "across_aeiemid" in modes:
#         last_out = _run_across_aeiemid(cfg)

#     if last_out is None:
#         raise ValueError("No run modes selected. Set run_modes: ['mobility_only', 'across_aeiemid'] in config.")
#     return last_out
