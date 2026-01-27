#!/usr/bin/env python3
import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ================== CONFIG ==================
IN_DIR = "./rq1_out/rating_decay_by_class"  # folder containing part-*.csv
OUT_FILE = "rq1_rating_decay.png"

# What to show
SHOW_CLASSES = {"viral", "traditional", "other"}
HIGHLIGHT = {"viral", "traditional"}                        # thicker line

# Crop window (matches your Spark output)
MIN_MSP = -6
MAX_MSP = 36

# Optional: hide points with tiny sample size (reduces noisy tail)
MIN_RATINGS_PER_POINT = 0  # e.g., 500 or 1000 if you want
# ===========================================


def find_part_csv(folder: str) -> str:
    files = sorted(glob.glob(os.path.join(folder, "part-*.csv")))
    if not files:
        raise FileNotFoundError(f"No part-*.csv found in {folder}")
    return files[0]


def read_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    path = find_part_csv(IN_DIR)
    rows = read_rows(path)

    # class -> month -> (avg_rating, n_ratings)
    data = defaultdict(dict)
    months_set = set()

    for r in rows:
        c = r.get("class")
        if c not in SHOW_CLASSES:
            continue

        try:
            m = int(r["months_since_peak"])
            n = int(float(r["n_ratings"]))
            a = float(r["avg_rating"])
        except Exception:
            continue

        if not (MIN_MSP <= m <= MAX_MSP):
            continue
        if n < MIN_RATINGS_PER_POINT:
            continue

        data[c][m] = (a, n)
        months_set.add(m)

    months = sorted(months_set)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot non-highlight first, highlight last
    classes = sorted(SHOW_CLASSES, key=lambda x: (x in HIGHLIGHT, x))

    for c in classes:
        y = []
        for m in months:
            if m in data[c]:
                y.append(data[c][m][0])
            else:
                y.append(None)  # break line if missing
        lw = 2.5 if c in HIGHLIGHT else 1.5
        ax.plot(months, y, marker="o", linewidth=lw, label=c)

    ax.set_title("Rating decay after peak attention (monthly alignment)")
    ax.set_xlabel("Months since peak interaction month")
    ax.set_ylabel("Average star rating")
    ax.grid(True, linewidth=0.3)
    ax.legend()

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
