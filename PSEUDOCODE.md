# Pseudocode — Women's mobility networks enable more efficient travel

Detailed description of the pipeline's functionality corresponding to the analysis in [De Sojo, Lehmann & Alessandretti (2026)](https://arxiv.org/abs/2604.00943).

---

```
PIPELINE: Women's mobility networks enable more efficient travel
================================================================

INPUT:
  - Stop-level mobility records (partitioned by user group)
  - User demographics (gender, country)
  - YAML configuration (Spark settings, paths, enabled metrics)

════════════════════════════════════════════════════
STAGE 1 — Compute mobility and network metrics  [scripts/01_compute_metrics.py]
════════════════════════════════════════════════════

FOR EACH user_group IN dataset:

  1. Load stops (loc, start, end, latitude, longitude)
  2. Filter invalid locations (loc IS NULL or loc = -1)
  3. Convert timestamps to local time; aggregate to monthly resolution
  4. Detect home / work anchors per user-month (HoWDe algorithm)
  5. Join gender and country from demographics; drop users with missing gender

  6. MOBILITY METRICS PIPELINE:
       FOR EACH (user, month):
         Compute visits, unique locations, radius of gyration,
         entropy, total stay time, return time, jump length
       Filter users with < 2 unique locations

  7. NETWORK METRICS PIPELINE:
       FOR EACH (user, month):
         Build directed mobility graph G = (V, E)
           V = unique locations visited
           E = sequential transitions between locations (weighted by frequency)
         Compute: degree, strength, clustering coefficient,
                  betweenness centrality, efficiency
       Optionally save origin-destination (OD) pairs

  8. Join mobility + network metrics; write per-group Parquet output

  9. IF get_allmet:
       Pool all user-groups into single DataFrame
       Assign activity deciles (visits quantiles 1–10)
       Assign repertoire deciles (locations quantiles 1–10)
       Label activity-repertoire groups: inactive / moderate / active
       Save as CSV

════════════════════════════════════════════════════
STAGE 2 — Gender differences in activity & repertoire  [scripts/02_analyze_gender_diff_Nk.py]
════════════════════════════════════════════════════

INPUT: pooled metrics CSV from Stage 1

FOR EACH country (and pooled sample):
  1. Compute visit (N) and location (k) distributions by gender
  2. Bootstrap median and standard error per gender
  3. Compute gender gap across N, k deciles
  4. Compute KS statistic (distributional distance M vs F)
  5. Bootstrap relative coefficient of variation (RCV) per gender
  OUTPUT: summary statistics → figures/fig1

════════════════════════════════════════════════════
STAGE 3 — Nearest-neighbour matching  [scripts/03_compute_nwmet_matching_byNk.py]
════════════════════════════════════════════════════

INPUT: pooled metrics CSV from Stage 1

FOR EACH country AND activity group:
  1. Standardise (N, k) features with z-score scaling
  2. For each female user, find nearest male neighbour in (N, k) space
  3. Compute observed gender difference Δ in each network metric
  4. Generate shuffled-reference distribution:
       REPEAT n_shuffles times:
         Randomly permute gender labels
         Recompute Δ_shuffled
  5. Significance: compare Δ_observed vs distribution of Δ_shuffled
  OUTPUT: matched pairs + observed vs shuffled Δ per metric

════════════════════════════════════════════════════
STAGE 4 — Post-process matched network results  [scripts/04_analyze_gender_diff_nwmet.py]
════════════════════════════════════════════════════

INPUT: matched results from Stage 3

1. Aggregate Δ across countries / activity groups
2. Anonymise user identifiers
3. Evaluate significance relative to shuffled baseline
OUTPUT: aggregated Δ tables → figures/fig2, fig4a

════════════════════════════════════════════════════
STAGE 5 — Compute movement sequences and tours  [scripts/05_compute_sequences_tours.py]
════════════════════════════════════════════════════

INPUT: stop-level records (same as Stage 1)

FOR EACH user_group IN dataset:

  1–5. Same preprocessing as Stage 1 (local time, HoWDe, demographics)
  6.   Replace location IDs with semantic labels (H = home, W = work)

  7. SEQUENCE CONSTRUCTION:
       FOR EACH (user, month):
         Order stops chronologically → stop_sequence [L1, L2, ..., Ln]
         Remove consecutive repeated locations
         Record arrival/departure times and positions (lat/lon)

  8. TOUR EXTRACTION:
       FOR EACH stop_sequence:
         Identify keystones: locations that anchor circular sub-trips
         Extract tours = sub-sequences that start and end at same keystone
         Assign main keystone for the monthly sequence

  9. EFFICIENCY METRICS (optional):
       FOR EACH stop_sequence:
         Compute pairwise distances between consecutive stops (seq_cost)
         Compute reward = max spatial spread achievable from anchor
         Compute efficiency = reward / cost
         Compute savings = reward − cost
         Anchor variants: trip start / home location / keystone

  10. TOUR SUMMARY:
        FOR EACH tour:
          Record: length (stops), total distance, reward,
                  max reward, duration
        Classify tours by length bin
        Merge activity-repertoire group from Stage 1

  OUTPUT: per-user-month tour dataset CSV, sequence-efficiency Parquet

════════════════════════════════════════════════════
STAGE 6 — Gender differences in tours  [scripts/06_analyze_gender_diff_tours.py]
════════════════════════════════════════════════════

INPUT: tour dataset CSV from Stage 5

FOR EACH country AND tour length bin:
  1. Compute tour length distribution by gender
  2. Bootstrap gender gap in tour counts per bin
  3. Compute pooled standard error across countries
  OUTPUT: gender gap statistics → figures/fig3

════════════════════════════════════════════════════
STAGE 7 — Gender differences in travel efficiency  [scripts/07_anlayze_gender_diff_toureff.py]
════════════════════════════════════════════════════

INPUT: sequence-efficiency dataset from Stage 5

FOR EACH country AND activity-repertoire group:
  1. Compute median efficiency (savings) by gender
  2. Bootstrap gender gap in efficiency across reward/cost regimes
  3. Stratify by activity quantile
  OUTPUT: efficiency contrast tables → figures/fig4

════════════════════════════════════════════════════
FIGURE GENERATION  [notebooks/]
════════════════════════════════════════════════════

fig1 ← Stage 2 outputs   (activity/repertoire distributions)
fig2 ← Stage 4 outputs   (matched network metric differences)
fig3 ← Stage 6 outputs   (tour length distributions)
fig4 ← Stage 4 + Stage 7 (efficiency and matched network results)
```
