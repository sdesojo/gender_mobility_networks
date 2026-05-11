
# Women's mobility networks enable more efficient travel

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

On a standard desktop computer with a stable internet connection, steps 3–4 take approximately 2 minutes (tested on macOS, Python 3.9.18).

---

## Demo

> **Note:** The results obtained from this demo dataset are not intended to reproduce the findings reported in the paper. The goal is solely to demonstrate that the code runs correctly and can be applied to a sample of users. To reproduce the quantitative results from the manuscript, see [Instructions for Use](#instructions-for-use).

### Sample data

A simulated dataset of 100 users can be generated using the preferential return model (`data/demo/generate_demo_data.py`). The generator produces stop-level records that match the pipeline's expected input format, targeting a median of ~80 total visits and ~20 unique locations per user-month, distributed across 10 countries.

Generate the demo data from the repository root:

```bash
python data/demo/generate_demo_data.py
```

This writes two files:
- `data/demo/stops/user_group=00/stops.parquet` — stop records with columns `useruuid`, `loc`, `start`, `end`, `latitude`, `longitude`, `timezone`
- `data/demo/demographics/demographics.parquet` — user attributes with columns `useruuid`, `gender`, `GID_0`, `NAME_0`

Requires: `numpy`, `pandas`, `pyarrow`. No Spark needed for generation.

### Running the demo

A ready-to-use configuration file is provided at `configs/demo_config.yaml`. Run the metrics pipeline on the demo data:

```bash
python scripts/01_compute_metrics.py --config configs/demo_config.yaml
```

### Expected output

The script writes:
- `data/demo/output/metrics/user_group=00/` — per-group Parquet tables with user-month-level mobility and network metrics
- `data/demo/output/user_metrics_demo.csv` — pooled metrics across all users, including activity/repertoire deciles and group labels

To inspect and validate the outputs interactively, open the notebook:

```bash
jupyter notebook data/demo/inspect_demo_data.ipynb
```

The notebook loads both the raw input data and the pipeline output (`user_metrics_demo.csv`), checks schema compatibility, and provides descriptive statistics and plots for all metric groups.

### Expected run time

On a standard desktop (4-core CPU, 8 GB RAM), generating the demo data takes under 10 seconds. Running `01_compute_metrics.py` on the 100-user demo dataset takes approximately 3-5 minutes with the default demo Spark configuration (2 workers, 4 GB memory).

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

To reproduce the quantitative results from the manuscript, follow the two-stage workflow below.

**Stage A — Metric computation** (requires raw stop-level data)

Raw stop-level mobility data are not publicly shared to preserve user anonymity. If you have access to equivalent stop-level traces, run the full pipeline:

```
compute metrics → statistical analyses → figure generation
```

PySpark settings (memory, workers, tmp directories) are read from the YAML configuration and should be tuned to your infrastructure. Scripts 03 and 06 include shuffled-reference procedures; run time depends on sample size and iteration counts.


**Stage B — Reproduce analyses from pre-computed data** (recommended)

The pre-computed metric datasets used in the paper are publicly available at:

> [https://doi.org/10.11583/DTU.31835038](https://doi.org/10.11583/DTU.31835038)

The repository contains two datasets:
- **User metrics** — user-month-level mobility and network metrics (output of `scripts/01_compute_metrics.py`)
- **Tour dataset** — identified tours and journey-level descriptors (output of `scripts/05_compute_sequences_tours.py`)

All metrics are described in detail in the Methods section of the paper. Download these datasets, set the `input_path` variable at the top of each script to point to the downloaded files, then run scripts 02–07 and the figure notebooks directly — skipping the PySpark computation steps entirely:

```bash
python scripts/02_analyze_gender_diff_Nk.py        # → Figure 1
python scripts/03_compute_nwmet_matching_byNk.py   # → Figures 2, 4a
python scripts/04_analyze_gender_diff_nwmet.py     # → Figure 2
python scripts/06_analyze_gender_diff_tours.py     # → Figure 3
python scripts/07_anlayze_gender_diff_toureff.py   # → Figure 4
jupyter notebook                                    # → open fig1–fig4 notebooks
```


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
