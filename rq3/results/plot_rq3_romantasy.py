#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from typing import List, Dict, Tuple, Any

import matplotlib.pyplot as plt


IN_BASE_PATH = "./romantasy_out/"
PUB_YEAR_PATH = "pub_year_trend_thresh_"
INTERACTION_YEAR_PATH = "interaction_year_trend_thresh_"
THRESH = 10

MIN_PLOT_YEAR = 2007
MAX_PLOT_YEAR = 2017

def find_single_csv(folder: str) -> str:
    # Spark output folder contains part-*.csv
    candidates = sorted(glob.glob(os.path.join(folder, "part-*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No part-*.csv found in {folder}")
    return candidates[0]


def read_csv_as_dicts(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(x: str):
    try:
        return int(x)
    except Exception:
        return None


def to_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rq3_romantasy_trends.png", help="Output image file")
    args = ap.parse_args()

    pub_csv = find_single_csv(IN_BASE_PATH + PUB_YEAR_PATH + str(THRESH))
    int_csv = find_single_csv(IN_BASE_PATH + INTERACTION_YEAR_PATH + str(THRESH))

    pub_rows = read_csv_as_dicts(pub_csv)
    int_rows = read_csv_as_dicts(int_csv)

    # Publication trend columns: pub_year, share_romantasy, n_books
    pub: List[Tuple[int, float, int]] = []
    for r in pub_rows:
        y = int(r.get("pub_year", ""))
        s = float(r.get("share_romantasy", ""))
        n = int(r.get("n_books", ""))
        if MIN_PLOT_YEAR <= y <= MAX_PLOT_YEAR:
            pub.append((y, s, n))
    pub.sort(key=lambda t: t[0])

    # Interaction trend columns: added_year, share_romantasy_interactions, n_interactions
    inter: List[Tuple[int, float, int]]  = []
    for r in int_rows:
        y = int(r.get("added_year", ""))
        s = float(r.get("share_romantasy_interactions", ""))
        n = int(r.get("n_interactions", ""))
        if MIN_PLOT_YEAR <= y <= MAX_PLOT_YEAR:
            inter.append((y, s, n))
    inter.sort(key=lambda t: t[0])

    if not pub:
        raise RuntimeError("No publication rows found after filtering. Adjust min/max year.")
    if not inter:
        raise RuntimeError("No interaction rows found after filtering. Adjust min/max year.")

    pub_years = [x[0] for x in pub]
    pub_share = [x[1] for x in pub]
    pub_n = [x[2] for x in pub]

    int_years = [x[0] for x in inter]
    int_share = [x[1] for x in inter]
    int_n = [x[2] for x in inter]

    # ---- Two-panel plot (best for slides) ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(pub_years, pub_share, marker="o", linewidth=1)
    ax1.set_title("Romantasy share among published books (supply)")
    ax1.set_ylabel("Share of romantasy books")
    ax1.grid(True, linewidth=0.3)

    ax2.plot(int_years, int_share, marker="o", linewidth=1)
    ax2.set_title("Romantasy share among user interactions (demand)")
    ax2.set_ylabel("Share of romantasy interactions")
    ax2.set_xlabel("Year")
    ax2.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
