"""
Generate simulated stop-level mobility data for demo purposes.

Simulates 100 users across 10 countries for one month (January 2025)
using the preferential return model. Visit counts follow a power-law
distribution targeting a median of ~80 total visits and ~20 unique
locations per user-month.

Run from the repository root:
    python data/demo/generate_demo_data.py

Output:
    data/demo/stops/user_group=00/stops.parquet
    data/demo/demographics/demographics.parquet
"""

import os
import numpy as np
import pandas as pd

# ---- Parameters ----
SEED = 42
N_USERS = 100

# January 2025 (UTC unix timestamps)
MONTH_START = 1735689600  # 2025-01-01 00:00:00 UTC
MONTH_END   = 1738368000  # 2025-02-01 00:00:00 UTC

# Preferential return model parameters (Pappalardo et al. 2015)
RHO   = 0.6
GAMMA = 0.21

COUNTRIES = {
    "JPN": {"name": "Japan",           "lat":  35.68,  "lon":  139.69, "tz":  32400},
    "GBR": {"name": "United Kingdom",  "lat":  51.51,  "lon":   -0.13, "tz":      0},
    "DEU": {"name": "Germany",         "lat":  52.52,  "lon":   13.40, "tz":   3600},
    "FRA": {"name": "France",          "lat":  48.86,  "lon":    2.35, "tz":   3600},
    "ESP": {"name": "Spain",           "lat":  40.42,  "lon":   -3.70, "tz":   3600},
    "TWN": {"name": "Taiwan",          "lat":  25.04,  "lon":  121.56, "tz":  28800},
    "NLD": {"name": "Netherlands",     "lat":  52.37,  "lon":    4.90, "tz":   3600},
    "AUS": {"name": "Australia",       "lat": -33.87,  "lon":  151.21, "tz":  36000},
    "USA": {"name": "United States",   "lat":  40.71,  "lon":  -74.01, "tz": -18000},
    "SWE": {"name": "Sweden",          "lat":  59.33,  "lon":   18.07, "tz":   3600},
}


def preferential_return_sequence(n_visits: int, rng: np.random.Generator) -> list:
    """
    Simulate a sequence of location visits using the preferential return model.

    At each step:
      - With probability rho * k^(-gamma), explore a new location.
      - Otherwise, return to a previously visited location with probability
        proportional to its past visit count.

    Returns a list of integer location IDs (0-indexed per user).
    """
    visits = [0]
    location_counts = {0: 1}

    for _ in range(n_visits - 1):
        k = len(location_counts)
        p_new = RHO * (k ** (-GAMMA))

        if rng.random() < p_new:
            new_loc = k
            location_counts[new_loc] = 1
            visits.append(new_loc)
        else:
            locs = list(location_counts.keys())
            counts = np.array([location_counts[l] for l in locs], dtype=float)
            probs = counts / counts.sum()
            chosen = int(rng.choice(locs, p=probs))
            location_counts[chosen] += 1
            visits.append(chosen)

    return visits


def generate_stops_for_user(
    useruuid: str,
    n_visits: int,
    country_info: dict,
    rng: np.random.Generator,
) -> list:
    """
    Generate stop records for a single user over one calendar month.

    Each stop has:
      - A location ID drawn from the preferential return sequence.
      - Coordinates sampled from a bivariate normal centred on the country capital.
      - Sequential timestamps; stop duration sampled from a log-normal.
    """
    sequence = preferential_return_sequence(n_visits, rng)
    n_locs = max(sequence) + 1

    # Assign stable coordinates to each location (clustered ~10–20 km spread)
    loc_lat = rng.normal(country_info["lat"], 0.15, size=n_locs)
    loc_lon = rng.normal(country_info["lon"], 0.20, size=n_locs)

    records = []
    current_time = int(MONTH_START + rng.integers(0, 3600))

    for loc_id in sequence:
        # Stop duration: log-normal, median ~30 min, clamped 5 min–4 h
        duration = int(rng.lognormal(mean=7.5, sigma=1.0))
        duration = max(300, min(duration, 14400))

        start = current_time
        end   = start + duration

        if end > MONTH_END:
            break

        records.append({
            "useruuid":  useruuid,
            "loc":       int(loc_id),
            "start":     int(start),
            "end":       int(end),
            "latitude":  float(round(loc_lat[loc_id], 6)),
            "longitude": float(round(loc_lon[loc_id], 6)),
            "timezone":  int(country_info["tz"]),
        })

        # Travel + idle gap between stops: log-normal, ~10–120 min
        gap = int(rng.lognormal(mean=7.0, sigma=0.8))
        gap = max(300, min(gap, 7200))
        current_time = end + gap

    return records


def main():
    rng = np.random.default_rng(SEED)

    country_codes = list(COUNTRIES.keys())
    genders = ["MALE", "FEMALE"]

    all_stops = []
    all_demo  = []

    for i in range(N_USERS):
        idx      = str(i + 1).zfill(3)
        useruuid = f"00-{idx}"
        gender   = rng.choice(genders)
        gid0     = rng.choice(country_codes)
        country_info = COUNTRIES[gid0]

        # Total visits: log-normal targeting median ~80, clamped 20–300
        n_visits = int(rng.lognormal(mean=4.38, sigma=0.4))
        n_visits = max(20, min(n_visits, 300))

        stops = generate_stops_for_user(useruuid, n_visits, country_info, rng)
        all_stops.extend(stops)

        all_demo.append({
            "useruuid": useruuid,
            "gender":   gender,
            "GID_0":    gid0,
            "NAME_0":   country_info["name"],
        })

    # ---- Save stops as Parquet ----
    df_stops = pd.DataFrame(
        all_stops,
        columns=["useruuid", "loc", "start", "end", "latitude", "longitude", "timezone"],
    )
    df_stops = df_stops.astype({
        "loc": "int64", "start": "int64", "end": "int64", "timezone": "int64",
        "latitude": "float64", "longitude": "float64",
    })

    stops_path = os.path.join(os.path.dirname(__file__), "stops", "user_group=00")
    os.makedirs(stops_path, exist_ok=True)
    df_stops.to_parquet(os.path.join(stops_path, "stops.parquet"), index=False)

    # ---- Save demographics as Parquet ----
    df_demo = pd.DataFrame(all_demo, columns=["useruuid", "gender", "GID_0", "NAME_0"])

    demo_path = os.path.join(os.path.dirname(__file__), "demographics")
    os.makedirs(demo_path, exist_ok=True)
    df_demo.to_parquet(os.path.join(demo_path, "demographics.parquet"), index=False)

    # ---- Summary ----
    summary = df_stops.groupby("useruuid").agg(
        total_visits=("loc", "count"),
        unique_locs=("loc", "nunique"),
    )
    print(f"Users generated:          {len(df_demo)}")
    print(f"Total stop records:       {len(df_stops)}")
    print(f"Median total visits:      {summary['total_visits'].median():.0f}")
    print(f"Median unique locations:  {summary['unique_locs'].median():.0f}")
    print(f"Stops saved to:           {stops_path}/")
    print(f"Demographics saved to:    {demo_path}/")


if __name__ == "__main__":
    main()
