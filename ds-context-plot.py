#!/usr/bin/env python3
import csv
import glob
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

IN_BASE = "./context_out"
OUT_FILE = "dataset_context.png"

MIN_YEAR = 1990
MAX_YEAR = 2017  # adjust as needed


def find_part_csv(folder: str) -> str:
    files = sorted(glob.glob(os.path.join(folder, "part-*.csv")))
    if not files:
        raise FileNotFoundError(f"No part-*.csv in {folder}")
    return files[0]


def read_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_series(folder: str, year_col: str, value_cols: list[str]):
    path = find_part_csv(folder)
    rows = read_rows(path)

    data = {c: {} for c in value_cols}
    years = set()

    for r in rows:
        try:
            y = int(r[year_col])
        except Exception:
            continue
        if not (MIN_YEAR <= y <= MAX_YEAR):
            continue
        years.add(y)
        for c in value_cols:
            try:
                data[c][y] = float(r[c])
            except Exception:
                data[c][y] = 0.0

    years = sorted(years)
    aligned = {c: [data[c].get(y, 0.0) for y in years] for c in value_cols}
    return years, aligned


def main():
    # Interactions
    years_i, vals_i = load_series(
        os.path.join(IN_BASE, "interactions_per_year"),
        year_col="year",
        value_cols=["n_interactions"],
    )

    # Publications
    years_p, vals_p = load_series(
        os.path.join(IN_BASE, "publications_per_year"),
        year_col="year",
        value_cols=["n_published_books"],
    )

    # Reviews
    years_r, vals_r = load_series(
        os.path.join(IN_BASE, "reviews_per_year"),
        year_col="year",
        value_cols=["n_reviews_total", "n_reviews_rating_only", "n_reviews_with_text"],
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    vals_p_k = [v / 1_000 for v in vals_p["n_published_books"]]
    ax1.plot(years_p, vals_p_k, marker="o", linewidth=1)
    ax1.set_title("Books per publication year")
    ax1.set_ylabel("Books published (thousands)")
    ax1.grid(True, linewidth=0.3)

    vals_i_m = [v / 1_000_000 for v in vals_i["n_interactions"]]
    ax2.plot(years_i, vals_i_m, marker="o", linewidth=1)
    ax2.set_title("Interactions per year (date_added)")
    ax2.set_ylabel("Interactions (millions)")
    ax2.grid(True, linewidth=0.3)

    vals_r_m = [v / 1_000_000 for v in vals_r["n_reviews_total"]]
    ax3.plot(years_r, vals_r_m, marker="o", linewidth=1)
    ax3.set_title("Reviews per year")
    ax3.set_ylabel("Reviews (millions)")
    ax3.set_xlabel("Year")
    ax3.grid(True, linewidth=0.3)

    # force integer x ticks
    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
