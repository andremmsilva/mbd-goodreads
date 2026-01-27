#!/usr/bin/env python3
from datetime import datetime
import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# -------- PATHS --------
PATH_INTERACTIONS = "/user/s3761576/goodreads/goodreads_interactions_dedup.parquet"
PATH_BOOKS = "/user/s3761576/goodreads/goodreads_books.parquet"
PATH_SENTIMENT = "/user/s3761576/sentiment_scores"  # from your colleague

OUT_BASE = "./rq2_out"

# -------- CONSTANTS --------
MIN_TOTAL = 1000
Q = 0.90
Q_LOW = 1.0 - Q

# filtering
EXCLUDE_RATING_0 = True
EXCLUDE_SENTIMENT_0 = True

# -------- HELPERS --------
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def month_num_from_abbrev(col):
    # date_added example: "Tue Mar 03 08:19:07 -0800 2015"
    m = F.regexp_extract(col, r"^[A-Za-z]{3}\s+([A-Za-z]{3})\s+\d{1,2}\s", 1)
    return (
        F.when(m == "Jan", 1).when(m == "Feb", 2).when(m == "Mar", 3)
         .when(m == "Apr", 4).when(m == "May", 5).when(m == "Jun", 6)
         .when(m == "Jul", 7).when(m == "Aug", 8).when(m == "Sep", 9)
         .when(m == "Oct", 10).when(m == "Nov", 11).when(m == "Dec", 12)
         .otherwise(F.lit(None))
    )

def yyyymm_to_index(col):
    year = F.floor(col / 100)
    month = col % 100
    return year * 12 + (month - 1)

# -------- MAIN --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_BASE)
    ap.add_argument("--min_total", type=int, default=MIN_TOTAL)
    ap.add_argument("--q", type=float, default=Q)
    args = ap.parse_args()

    out_base = args.out.rstrip("/")
    min_total = args.min_total
    q = args.q
    q_low = 1.0 - q

    spark = SparkSession.builder.appName("rq2_sentiment_vs_rating_by_class").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------
    # 1) Build virality classes from interactions (same as your RQ1)
    # ------------------------------------------------------------
    log("1) Reading interactions and extracting YYYYMM...")
    inter = (
        spark.read.parquet(PATH_INTERACTIONS)
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

    log("2) Monthly counts per book...")
    monthly = inter.groupBy("book_id", "yyyymm").agg(F.count("*").alias("n_month"))

    log("3) Per-book totals and first active month...")
    totals = (
        monthly.groupBy("book_id")
               .agg(
                   F.sum("n_month").alias("total_interactions"),
                   F.min("yyyymm").alias("first_yyyymm"),
               )
               .where(F.col("total_interactions") >= F.lit(min_total))
    )

    log("4) Peak month per book...")
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

    log("5) Compute peak_share and time_to_peak...")
    feats = (
        totals.join(peak, on="book_id", how="inner")
              .withColumn("peak_share", F.col("peak_interactions") / F.col("total_interactions"))
              .withColumn("first_idx", yyyymm_to_index(F.col("first_yyyymm")))
              .withColumn("peak_idx", yyyymm_to_index(F.col("peak_yyyymm")))
              .withColumn("time_to_peak_months", (F.col("peak_idx") - F.col("first_idx")).cast("int"))
              .drop("first_idx", "peak_idx")
              .select("book_id", "peak_share", "time_to_peak_months")
    )

    log("6) Quantile thresholds (data-driven)...")
    ps_low, ps_high = feats.approxQuantile("peak_share", [q_low, q], 0.001)
    ttp_low, ttp_high = feats.approxQuantile("time_to_peak_months", [q_low, q], 0.001)

    log(f"   -> MIN_TOTAL={min_total}, q={q}")
    log(f"   -> peak_share: p{int(q_low*100)}={ps_low:.4f}, p{int(q*100)}={ps_high:.4f}")
    log(f"   -> time_to_peak_months: p{int(q_low*100)}={ttp_low:.2f}, p{int(q*100)}={ttp_high:.2f}")

    log("7) Label classes (viral / traditional / other)...")
    labeled = (
        feats.withColumn(
            "class",
            F.when((F.col("peak_share") >= F.lit(ps_high)) & (F.col("time_to_peak_months") <= F.lit(ttp_low)), F.lit("viral"))
             .when((F.col("peak_share") <= F.lit(ps_low)) & (F.col("time_to_peak_months") >= F.lit(ttp_high)), F.lit("traditional"))
             .otherwise(F.lit("other"))
        )
        .select("book_id", "class")
        .persist()
    )

    class_counts = labeled.groupBy("class").count().orderBy("class")
    log("   -> Class counts:")
    class_counts.show(truncate=False)

    # ------------------------------------------------------------
    # 2) Load sentiment table and clean
    # ------------------------------------------------------------
    log("8) Loading sentiment_scores parquet...")
    sent = (
        spark.read.parquet(PATH_SENTIMENT)
        .select(
            F.col("book_id").cast("long").alias("book_id"),
            F.col("rating").cast("int").alias("rating"),
            F.col("sentiment_score").cast("double").alias("sentiment_score"),
        )
        .where(F.col("book_id").isNotNull() & F.col("rating").isNotNull() & F.col("sentiment_score").isNotNull())
    )

    # Filter rating range (Goodreads should be 0-5 in your data)
    sent = sent.where((F.col("rating") >= 0) & (F.col("rating") <= 5))

    if EXCLUDE_RATING_0:
        sent = sent.where(F.col("rating") != 0)
    if EXCLUDE_SENTIMENT_0:
        sent = sent.where(F.col("sentiment_score") != 0.0)

    # ------------------------------------------------------------
    # 3) Join with class labels
    # ------------------------------------------------------------
    log("9) Joining sentiment rows with class labels...")
    joined = sent.join(labeled, on="book_id", how="inner").select("class", "rating", "sentiment_score")

    # ------------------------------------------------------------
    # 4) Output A: sentiment distribution per class and star rating
    # ------------------------------------------------------------
    log("10) Aggregating sentiment by (class, rating)...")
    by_class_rating = (
        joined.groupBy("class", "rating")
        .agg(
            F.count("*").alias("n"),
            F.avg("sentiment_score").alias("avg_sentiment"),
            F.expr("percentile_approx(sentiment_score, 0.05)").alias("p05_sentiment"),
            F.expr("percentile_approx(sentiment_score, 0.95)").alias("p95_sentiment"),
        )
        .orderBy("class", "rating")
    )

    out_a = f"{out_base}/sentiment_by_class_rating_min{min_total}_q{q}"
    by_class_rating.coalesce(1).write.mode("overwrite").option("header", True).csv(out_a)
    log(f"   -> wrote {out_a}")
    by_class_rating.show(50, truncate=False)

    # ------------------------------------------------------------
    # 5) Output B: correlation rating<->sentiment per class
    # ------------------------------------------------------------
    log("11) Computing correlation per class...")
    corr_by_class = (
        joined.groupBy("class")
        .agg(
            F.count("*").alias("n"),
            F.corr("rating", "sentiment_score").alias("pearson_corr"),
        )
        .orderBy("class")
    )

    out_b = f"{out_base}/corr_by_class_min{min_total}_q{q}"
    corr_by_class.coalesce(1).write.mode("overwrite").option("header", True).csv(out_b)
    log(f"   -> wrote {out_b}")
    corr_by_class.show(truncate=False)

    labeled.unpersist()
    spark.stop()
    log("DONE.")

if __name__ == "__main__":
    main()

