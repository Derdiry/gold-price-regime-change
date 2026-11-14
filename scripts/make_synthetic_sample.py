"""
Write data/raw/new_Data_SYNTHETIC_sample.csv: 20 SYNTHETIC rows in the column
schema of the private Sudan file data/raw/new_Data.csv (not in this repository).

No value comes from the real file and no relationship between columns is
modelled: every numeric column is an independent seeded random walk. The rows
show the file's format only. The dates (January 2000) lie outside the real
sample (2015-01-06 -> 2022-12-22), so they cannot be mistaken for observations.

    python scripts/make_synthetic_sample.py
"""

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/new_Data_SYNTHETIC_sample.csv"
HEADER = ["date", "Price", "P/O 24", "p/g 21", "local USD", "Open", "High", "Low", "vol"]
N_ROWS = 20
SEED = 20000103
# (start value, daily step standard deviation, decimals) for each numeric column, chosen arbitrarily
WALKS = {"Price": (500.0, 2.0, 1), "P/O 24": (1000.0, 5.0, 1), "p/g 21": (45.0, 0.2, 2),
         "local USD": (10.0, 0.05, 2), "Open": (1000.0, 5.0, 1), "vol": (200.0, 20.0, 2)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    days, d = [], date(2000, 1, 3)
    while len(days) < N_ROWS:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    cols = {name: np.round(start + np.cumsum(rng.normal(0, sd, N_ROWS)), dec)
            for name, (start, sd, dec) in WALKS.items()}
    high = np.round(cols["Open"] + np.abs(rng.normal(0, 3, N_ROWS)), 1)
    low = np.round(cols["Open"] - np.abs(rng.normal(0, 3, N_ROWS)), 1)
    vol = np.abs(cols["vol"])
    with OUT.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for i, day in enumerate(days):
            w.writerow([f"{day.day}-{day.strftime('%b-%Y')}", cols["Price"][i], cols["P/O 24"][i], cols["p/g 21"][i],
                        cols["local USD"][i], cols["Open"][i], high[i], low[i], vol[i]])
    print(f"wrote {OUT.relative_to(ROOT)} ({N_ROWS} synthetic rows)")


if __name__ == "__main__":
    main()
