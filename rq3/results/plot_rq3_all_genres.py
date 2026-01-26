#!/usr/bin/env python3
import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ================== CONFIG ==================
IN_BASE_PATH = "./genre_out/"
THRESH = 10

PUB_DIR = f"pub_year_genre_shares_thresh_{THRESH}"
INT_DIR = f"interaction_year_genre_shares_thresh_{THRESH}"

OUT_FILE = "rq3_genre_comparison.png"

# Change these:
K_TOP = 5              # plot top-K genres + romantasy
MIN_YEAR = 2007         # crop 
MAX_YEAR = 2017         

HIGHLIGHT_GENRE = "romantasy"
# ===========================================


def find_single_csv(folder: str) -> str:
    candidates = sorted(glob.glob(os.path.join(folder, "part-*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No part-*.csv found in {folder}")
    return candidates[0]


def read_csv_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_top_k_genres(rows, year_key, share_key, min_year, max_year, k):
    """
    Pick top-k genres by mean share over [min_year, max_year].
    Returns a set of genres.
    """
    sums = defaultdict(float)
    cnts = defaultdict(int)

    for r in rows:
        try:
            year = int(r[year_key])
            genre = r["genre"]
            share = float(r[share_key])
        except Exception:
            continue

        if min_year <= year <= max_year:
            sums[genre] += share
            cnts[genre] += 1

    means = []
    for g in sums:
        if cnts[g] > 0:
            means.append((sums[g] / cnts[g], g))

    means.sort(reverse=True)
    return {g for _, g in means[:k]}


def build_series(rows, year_key, share_key, min_year, max_year, genres_to_plot):
    """
    Build series dict: genre -> {year -> share}
    """
    series = {g: {} for g in genres_to_plot}
    years_set = set()

    for r in rows:
        try:
            year = int(r[year_key])
            genre = r["genre"]
            share = float(r[share_key])
        except Exception:
            continue

        if genre not in series:
            continue
        if not (min_year <= year <= max_year):
            continue

        series[genre][year] = share
        years_set.add(year)

    years = sorted(years_set)
    # Ensure each genre has y-values aligned to same x-axis; missing -> None
    aligned = {}
    for g in genres_to_plot:
        aligned[g] = [series[g].get(y, None) for y in years]

    return years, aligned


def plot_panel(ax, years, aligned, title, ylabel):
    # Plot non-highlight first, highlight last so it sits on top.
    genres = sorted(aligned.keys(), key=lambda g: (g == HIGHLIGHT_GENRE, g))

    for g in genres:
        y = aligned[g]
        # matplotlib will break lines on None -> good for missing early years
        if g == HIGHLIGHT_GENRE:
            ax.plot(years, y, marker="o", linewidth=3, label=g)
        else:
            ax.plot(years, y, marker="o", linewidth=0.9, label=g)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3)
    ax.legend(ncol=2, fontsize=8)


def main():
    pub_csv = find_single_csv(os.path.join(IN_BASE_PATH, PUB_DIR))
    int_csv = find_single_csv(os.path.join(IN_BASE_PATH, INT_DIR))

    pub_rows = read_csv_rows(pub_csv)
    int_rows = read_csv_rows(int_csv)

    # Choose top K separately for supply and demand (often slightly different).
    top_pub = pick_top_k_genres(
        pub_rows, year_key="pub_year", share_key="share_books_genre",
        min_year=MIN_YEAR, max_year=MAX_YEAR, k=K_TOP
    )
    top_int = pick_top_k_genres(
        int_rows, year_key="added_year", share_key="share_interactions_genre",
        min_year=MIN_YEAR, max_year=MAX_YEAR, k=K_TOP
    )

    # Always include romantasy
    top_pub.add(HIGHLIGHT_GENRE)
    top_int.add(HIGHLIGHT_GENRE)

    # Build aligned series
    pub_years, pub_aligned = build_series(
        pub_rows, year_key="pub_year", share_key="share_books_genre",
        min_year=MIN_YEAR, max_year=MAX_YEAR, genres_to_plot=top_pub
    )
    int_years, int_aligned = build_series(
        int_rows, year_key="added_year", share_key="share_interactions_genre",
        min_year=MIN_YEAR, max_year=MAX_YEAR, genres_to_plot=top_int
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

    plot_panel(
        ax1, pub_years, pub_aligned,
        title=f"Genre share among published books (supply) — top {K_TOP} + romantasy (THRESH={THRESH})",
        ylabel="Share of books"
    )
    plot_panel(
        ax2, int_years, int_aligned,
        title=f"Genre share among user interactions (demand) — top {K_TOP} + romantasy (THRESH={THRESH})",
        ylabel="Share of interactions"
    )

    ax2.set_xlabel("Year")

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
