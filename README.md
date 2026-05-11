
# Women's mobility networks enable more efficient travel

## Contents

- [Overview](#overview)
- [Repository Organization](#repository-organization)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Instructions for Use](#instructions-for-use)
- [License](#license)
- [Citation](#citation)

---

## Overview

This repository contains the full computational pipeline for:

**[Women's mobility networks enable more efficient travel](https://arxiv.org/abs/2604.00943)**
De Sojo, Lehmann & Alessandretti — arXiv:2604.00943 (2026)

We model individual movement as a network of sequential location visits, constructed from smartphone traces linked to self-reported gender. The central finding is that women's mobility networks are simultaneously more clustered and more home-anchored than men's — a signature of trip-chaining, the practice of combining multiple errands into a single outing. This structural difference translates directly into greater travel efficiency: women cover more unique destinations per unit of distance traveled, on average, when measured over monthly movement sequences.

The pipeline quantifies these differences along four dimensions:

- **Activity and repertoire** — visit counts and unique-location diversity per user-month.
- **Mobility-network topology** — centrality and graph-level observables derived from individual mobility networks.
- **Tour and sequence organization** — how stops are chained into trips and the role of keystone locations.
- **Travel efficiency** — a distance-based metric balancing spatial reward against travel cost, evaluated under both matched and unmatched comparisons.

The analysis runs in three stages: PySpark pipelines compute metrics from raw stop-level traces; Python scripts perform statistical analyses; Jupyter notebooks generate the publication figures. Each stage is self-contained and can be run independently given the appropriate inputs.

For a complete, detailed description of the code's functionality, see [PSEUDOCODE.md](PSEUDOCODE.md).

---

## System Requirements

### Software dependencies

| Package | Version tested |
|---|---|
| Python | 3.9.18 |
| Java (JDK) | OpenJDK 20.0.2 |
| PySpark | 3.3.2 |
| pandas | 2.1.1 |
| numpy | 1.26.0 |
| scipy | 1.12.0 |
| scikit-learn | 1.3.0 |
| networkx | 3.1 |
| matplotlib | 3.8.0 |
| tqdm | 4.66.1 |
| pyyaml | 6.0.1 |
| howde | 2.0.0 |

### Operating systems tested

Debian GNU/Linux 12 (bookworm), x86\_64

### Non-standard hardware

The PySpark pipeline scripts (`01_compute_metrics.py`, `05_compute_sequences_tours.py`) were run on a compute cluster. Running them on a standard desktop is possible but will be significantly slower depending on data volume. No GPU is required.

---

## Installation Guide

### Instructions

1. Clone the repository:

```bash
git clone https://github.com/sdesojo/gender_mobility_networks.git
cd gender_mobility_networks
```

2. Install Java (required for PySpark). On Ubuntu:

```bash
sudo apt-get install default-jdk
```

On macOS:

```bash
brew install openjdk
```

3. Install the Python package and its dependencies:

```bash
pip install -e .
```

This installs the declared dependencies (`pyspark`, `pyyaml`, `tqdm`). For the full analysis pipeline also install:

```bash
pip install pandas numpy scipy scikit-learn networkx matplotlib jupyter
```

4. Install `howde` (home/work detection library):

```bash
pip howde
```

### Typical install time

On a standard desktop computer with a stable internet connection, steps 3–4 take approximately 5–10 minutes.

---

## Demo

### Sample data

<!-- TODO: is there a small sample dataset available for demo purposes? If so, describe it and where to find it. If not, describe the minimum input format needed. -->

The pipeline expects:

- Stop-level mobility records partitioned by user group (Parquet format recommended).
- User-level demographic attributes including gender and country.
- A YAML configuration file (see `configs/monthly_met_all_config.yaml` as a template).

### Running the demo

Update the paths in `configs/monthly_met_all_config.yaml` to point to your data, then run:

```bash
python scripts/01_compute_metrics.py --config configs/monthly_met_all_config.yaml
```

To run for a single user group:

```bash
python scripts/01_compute_metrics.py --config configs/monthly_met_all_config.yaml --user_group user_group=00
```

### Expected output

The script writes per-group Parquet tables containing user-month-level mobility and network metrics, and optionally pooled tables for downstream analyses.

### Expected run time

<!-- TODO: add approximate run time for the demo/sample data on a normal desktop -->

Run time on the full dataset is several hours per monthly batch on a compute cluster. Run time scales with data volume and PySpark configuration (memory, number of workers).

---

## Instructions for Use

### How to run on your data

1. Prepare stop-level mobility data and demographic attributes in the expected format.
2. Create a YAML configuration file (based on `configs/monthly_met_all_config.yaml`) specifying input/output paths, Spark resources, temporal aggregation, and enabled metrics.
3. Run the pipeline scripts in order:

```bash
# Step 1 — compute mobility and network metrics
python scripts/01_compute_metrics.py --config configs/<your_config>.yaml

# Step 2 — analyze gender differences in activity/repertoire (Figure 1)
python scripts/02_analyze_gender_diff_Nk.py

# Step 3 — matched comparison for network metrics (Figures 2 and 4a)
python scripts/03_compute_nwmet_matching_byNk.py

# Step 4 — post-process matched network results (Figure 2)
python scripts/04_analyze_gender_diff_nwmet.py

# Step 5 — compute movement sequences and tours
python scripts/05_compute_sequences_tours.py --config configs/<your_config>.yaml

# Step 6 — analyze gender differences in tours (Figure 3)
python scripts/06_analyze_gender_diff_tours.py

# Step 7 — analyze gender differences in travel efficiency (Figure 4)
python scripts/07_anlayze_gender_diff_toureff.py
```

Scripts 02–04 and 06–07 expose editable `input_path` / `output_path` variables at the top of the file — update these to match your local paths before running.

4. Generate figures using the notebooks in `notebooks/`:

```bash
jupyter notebook
```

Open the relevant notebook (`fig1` to `fig4`) and run all cells.

### Reproduction instructions

To reproduce the quantitative results from the manuscript, run all seven scripts in order on the original dataset, then open and execute the figure notebooks. The intended execution model is strictly staged:

```
compute metrics → statistical analyses → figure generation
```

Note:
- Scripts 03 and 06 include shuffled-reference procedures; run time depends on sample size and iteration counts.
- PySpark settings (memory, workers, tmp directories) are read from the YAML configuration and should be tuned to your infrastructure.

---

## Repository Organization

- `src/compute_metrics/`: PySpark modules to compute mobility and network metrics from stop-level mobility data.
- `src/compute_seq_tours/`: PySpark modules to build sequences/tours and derive sequence-level efficiency descriptors.
- `src/nearest_neighbor_matching/`: nearest-neighbor matching utilities to compare users under controlled activity/repertoire conditions.
- `src/statistical_analysis/`: statistical helper functions (bootstrap summaries, distributional comparisons, quantile analyses).
- `scripts/`: executable analysis scripts corresponding to the main stages of the paper.
- `configs/`: YAML configuration files controlling Spark resources, temporal aggregation, enabled metrics, and input/output paths.
- `notebooks/`: figure-generation notebooks (`fig1` to `fig4`).
- `figures/`: exported figure outputs.

---

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@article{de2026women,
  title={Women's mobility networks enable more efficient travel},
  author={de Sojo, S{\'\i}lvia and Lehmann, Sune and Alessandretti, Laura},
  journal={arXiv preprint arXiv:2604.00943},
  year={2026}
}
```

---

## License

MIT License.
