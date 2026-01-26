#!/usr/bin/env python3
from datetime import datetime
import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# ---- Defaults ----
DEFAULT_INTERACTIONS = "/user/s3761576/goodreads/goodreads_interactions_dedup.parquet"
DEFAULT_BOOKS = "/user/s3761576/goodreads/goodreads_books.parquet"

# ---- Helpers ----
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def month_num_from_abbrev(col):
    # date_added looks like: "Tue Mar 03 08:19:07 -0800 2015"
    # extract month abbrev: Mar, Apr, ...
    m = F.regexp_extract(col, r"^[A-Za-z]{3}\s+([A-Za-z]{3})\s+\d{1,2}\s", 1)
    return (
        F.when(m == "Jan", F.lit(1))
         .when(m == "Feb", F.lit(2))
         .when(m == "Mar", F.lit(3))
         .when(m == "Apr", F.lit(4))
         .when(m == "May", F.lit(5))
         .when(m == "Jun", F.lit(6))
         .when(m == "Jul", F.lit(7))
         .when(m == "Aug", F.lit(8))
         .when(m == "Sep", F.lit(9))
         .when(m == "Oct", F.lit(10))
         .when(m == "Nov", F.lit(11))
         .when(m == "Dec", F.lit(12))
         .otherwise(F.lit(None))
    )

def yyyymm_to_index(yyyymm_col):
    # yyyymm -> integer month index for deltas: year*12 + (month-1)
    year = F.floor(yyyymm_col / 100)
    month = yyyymm_col % 100
    return year * 12 + (month - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", default=DEFAULT_INTERACTIONS)
    ap.add_argument("--books", default=DEFAULT_BOOKS)
    ap.add_argument("--min_total", type=int, default=1000, help="min total interactions per book")
    ap.add_argument("--q", type=float, default=0.90, help="quantile for extremes (0.90 means 90/10 split)")
    ap.add_argument("--max_print", type=int, default=10)
    args = ap.parse_args()

    spark = SparkSession.builder.appName("virality_metrics_monthly").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    MIN_TOTAL = args.min_total
    Q = args.q
    Q_LOW = 1.0 - Q
    MAX_PRINT = args.max_print

    log("1) Reading interactions and extracting YYYYMM...")
    inter = (
        spark.read.parquet(args.interactions)
        .select(F.col("book_id").cast("long").alias("book_id"), "date_added")
        .where(F.col("date_added").isNotNull() & (F.length("date_added") > 0))
    )

    year = F.regexp_extract(F.col("date_added"), r"(\d{4})$", 1).cast("int")
    month = month_num_from_abbrev(F.col("date_added"))

    inter = (
        inter.withColumn("year", year)
             .withColumn("month", month)
             .where(F.col("year").isNotNull() & F.col("month").isNotNull())
             .withColumn("yyyymm", F.col("year") * 100 + F.col("month"))
             .select("book_id", "yyyymm")
    )

    log("2) Monthly counts per book: groupBy(book_id, yyyymm)...")
    monthly = (
        inter.groupBy("book_id", "yyyymm")
             .agg(F.count("*").alias("n_month"))
    )

    log("3) Per-book totals and first active month...")
    totals = (
        monthly.groupBy("book_id")
               .agg(
                   F.sum("n_month").alias("total_interactions"),
                   F.min("yyyymm").alias("first_yyyymm"),
               )
               .where(F.col("total_interactions") >= F.lit(MIN_TOTAL))
    )

    log("4) Peak month per book (max n_month, tie -> earliest month)...")
    w_peak = Window.partitionBy("book_id").orderBy(F.col("n_month").desc(), F.col("yyyymm").asc())
    peak = (
        monthly.join(totals.select("book_id"), on="book_id", how="inner")
               .withColumn("rn", F.row_number().over(w_peak))
               .where(F.col("rn") == 1)
               .select(
                   "book_id",
                   F.col("yyyymm").alias("peak_yyyymm"),
                   F.col("n_month").alias("peak_interactions"),
               )
    )

    log("5) Compute peak_share and time_to_peak (months)...")
    feats = (
        totals.join(peak, on="book_id", how="inner")
              .withColumn("peak_share", F.col("peak_interactions") / F.col("total_interactions"))
              .withColumn("first_idx", yyyymm_to_index(F.col("first_yyyymm")))
              .withColumn("peak_idx", yyyymm_to_index(F.col("peak_yyyymm")))
              .withColumn("time_to_peak_months", (F.col("peak_idx") - F.col("first_idx")).cast("int"))
              .drop("first_idx", "peak_idx")
    )

    log("6) Choose thresholds from quantiles (data-driven)...")
    # approxQuantile returns lists
    ps_low, ps_high = feats.approxQuantile("peak_share", [Q_LOW, Q], 0.001)
    ttp_low, ttp_high = feats.approxQuantile("time_to_peak_months", [Q_LOW, Q], 0.001)

    log(f"   -> Using MIN_TOTAL={MIN_TOTAL}")
    log(f"   -> peak_share quantiles: p{int(Q_LOW*100)}={ps_low:.4f}, p{int(Q*100)}={ps_high:.4f}")
    log(f"   -> time_to_peak_months quantiles: p{int(Q_LOW*100)}={ttp_low:.2f}, p{int(Q*100)}={ttp_high:.2f}")

    log("7) Classify books...")
    feats = feats.withColumn(
        "class",
        F.when((F.col("peak_share") >= F.lit(ps_high)) & (F.col("time_to_peak_months") <= F.lit(ttp_low)), F.lit("viral"))
         .when((F.col("peak_share") <= F.lit(ps_low)) & (F.col("time_to_peak_months") >= F.lit(ttp_high)), F.lit("traditional"))
         .otherwise(F.lit("other"))
    )

    log("8) Join titles for printing...")
    books = spark.read.parquet(args.books).select(F.col("book_id").cast("long").alias("book_id"), "title")
    feats_t = feats.join(books, on="book_id", how="left")

    log("9) Print TOP viral examples (highest peak_share, fastest time_to_peak)...")
    viral_top = (
        feats_t.where(F.col("class") == "viral")
               .orderBy(F.col("peak_share").desc(), F.col("time_to_peak_months").asc(), F.col("total_interactions").desc())
               .select("book_id", "title", "total_interactions", "peak_yyyymm", "peak_interactions", "peak_share", "time_to_peak_months")
               .limit(MAX_PRINT)
    )
    viral_top.show(MAX_PRINT, truncate=False)

    log("10) Print TOP traditional examples (lowest peak_share, slowest time_to_peak)...")
    trad_top = (
        feats_t.where(F.col("class") == "traditional")
               .orderBy(F.col("peak_share").asc(), F.col("time_to_peak_months").desc(), F.col("total_interactions").desc())
               .select("book_id", "title", "total_interactions", "peak_yyyymm", "peak_interactions", "peak_share", "time_to_peak_months")
               .limit(MAX_PRINT)
    )
    trad_top.show(MAX_PRINT, truncate=False)

    log("DONE.")
    spark.stop()


if __name__ == "__main__":
    main()
