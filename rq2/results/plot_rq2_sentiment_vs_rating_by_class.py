#!/usr/bin/env python3
import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===================== CONFIG =====================
IN_DIR = "./rq2_out/sentiment_by_class_rating_min1000_q0.9"
OUT_FILE = "rq2_sentiment_vs_rating_by_class.png"

# Which classes to plot and in what order
CLASSES = ["traditional", "viral", "other"]   # reorder as you like
HIGHLIGHT = "viral"                          # thicker line

# Filter / display
RATINGS = [1, 2, 3, 4, 5]
MIN_N_PER_POINT = 0                          # e.g., 500 to drop tiny groups

TITLE = "RQ2: Sentiment vs star rating by book class"
Y_LABEL = "Average review sentiment (VADER)"
X_LABEL = "Star rating"
# ==================================================


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

    # data[class][rating] = (avg, p05, p95, n)
    data = defaultdict(dict)

    for r in rows:
        c = r.get("class")
        if c not in set(CLASSES):
            continue

        try:
            rating = int(float(r["rating"]))
            n = int(float(r["n"]))
            avg = float(r["avg_sentiment"])
            p05 = float(r["p05_sentiment"])
            p95 = float(r["p95_sentiment"])
        except Exception:
            continue

        if rating not in RATINGS:
            continue
        if n < MIN_N_PER_POINT:
            continue

        data[c][rating] = (avg, p05, p95, n)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    for c in CLASSES:
        xs, ys, yerr_low, yerr_high = [], [], [], []
        for rating in RATINGS:
            if rating not in data[c]:
                continue
            avg, p05, p95, n = data[c][rating]
            xs.append(rating)
            ys.append(avg)
            yerr_low.append(avg - p05)
            yerr_high.append(p95 - avg)

        if not xs:
            continue

        lw = 2.8 if c == HIGHLIGHT else 1.6
        ax.errorbar(
            xs, ys,
            yerr=[yerr_low, yerr_high],
            marker="o",
            linewidth=lw,
            capsize=4,
            label=c
        )

    ax.set_title(TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    # ensure x ticks are 1..5 as integers
    ax.set_xticks(RATINGS)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, linewidth=0.3)
    ax.legend(title="Book class")

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
