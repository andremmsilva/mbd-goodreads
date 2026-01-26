#!/usr/bin/env python3
from datetime import datetime
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# -------- PATHS --------
PATH_INTERACTIONS = "/user/s3761576/goodreads/goodreads_interactions_dedup.parquet"
PATH_REVIEWS = "/user/s3761576/goodreads/goodreads_reviews_dedup.parquet"
PATH_BOOKS = "/user/s3761576/goodreads/goodreads_books.parquet"

OUT_PATH = "./rq1_out/rating_decay_by_class"

# -------- CONSTANTS --------
MIN_TOTAL = 1000
Q = 0.90
Q_LOW = 1.0 - Q
MAX_PRINT = 10

# -------- HELPERS --------
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def month_num_from_abbrev(col):
    m = F.regexp_extract(col, r"^[A-Za-z]{3}\s+([A-Za-z]{3})\s+\d{1,2}\s", 1)
    return (
        F.when(m == "Jan", 1).when(m == "Feb", 2).when(m == "Mar", 3)
         .when(m == "Apr", 4).when(m == "May", 5).when(m == "Jun", 6)
         .when(m == "Jul", 7).when(m == "Aug", 8).when(m == "Sep", 9)
         .when(m == "Oct", 10).when(m == "Nov", 11).when(m == "Dec", 12)
    )

def yyyymm_to_index(col):
    year = F.floor(col / 100)
    month = col % 100
    return year * 12 + (month - 1)

# -------- MAIN --------
def main():
    spark = SparkSession.builder.appName("rq1_rating_decay").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")


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
    books = spark.read.parquet(PATH_BOOKS).select(F.col("book_id").cast("long").alias("book_id"), "title")
    feats_t = feats.join(books, on="book_id", how="left")

    # ---- 2) Load reviews with ratings ----
    log("9) Loading reviews with ratings...")
    reviews = (
        spark.read.parquet(PATH_REVIEWS)
        .select(
            F.col("book_id").cast("long").alias("book_id"),
            "date_added",
            F.col("rating").cast("double")
        )
        .where(F.col("rating").isNotNull())
    )

    year = F.regexp_extract(F.col("date_added"), r"(\d{4})$", 1).cast("int")
    month = month_num_from_abbrev(F.col("date_added"))

    reviews = (
        reviews.withColumn("year", year)
               .withColumn("month", month)
               .where(F.col("year").isNotNull() & F.col("month").isNotNull())
               .withColumn("yyyymm", F.col("year") * 100 + F.col("month"))
               .select("book_id", "yyyymm", "rating")
    )

    # ---- 3) Join reviews with virality labels ----
    log("10) Joining reviews with virality class...")
    joined = (
        reviews.join(feats.select("book_id", "peak_yyyymm", "class"),
                     on="book_id", how="inner")
    )

    # ---- 4) Compute months since peak ----
    joined = (
        joined
        .withColumn("review_idx", yyyymm_to_index(F.col("yyyymm")))
        .withColumn("peak_idx", yyyymm_to_index(F.col("peak_yyyymm")))
        .withColumn("months_since_peak",
                    (F.col("review_idx") - F.col("peak_idx")).cast("int"))
        .where((F.col("months_since_peak") >= -6) & (F.col("months_since_peak") <= 36))
    )

    # ---- 5) Aggregate by class and months_since_peak ----
    log("11) Aggregating ratings over time since peak...")
    rating_decay = (
        joined.groupBy("class", "months_since_peak")
        .agg(
            F.count("*").alias("n_ratings"),
            F.avg("rating").alias("avg_rating")
        )
        .orderBy("class", "months_since_peak")
    )

    rating_decay.coalesce(1).write.mode("overwrite").option("header", True).csv(OUT_PATH)
    log(f"DONE. Output written to {OUT_PATH}")

    spark.stop()

if __name__ == "__main__":
    main()
