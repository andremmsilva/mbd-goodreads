#!/usr/bin/env python3
from datetime import datetime
import argparse

from pyspark.sql import SparkSession, functions as F

# ----------------------------
# Defaults (can be overridden by CLI args)
# ----------------------------
DEFAULT_BOOKS_PARQUET = "/user/s3761576/goodreads/goodreads_books.parquet"
DEFAULT_INTERACTIONS_PARQUET = "/user/s3761576/goodreads/goodreads_interactions_dedup.parquet"
DEFAULT_REVIEWS_PARQUET = "/user/s3761576/goodreads/goodreads_reviews_dedup.parquet"

DEFAULT_OUT_BASE = "./context_out"

# sanity range for plotting/context (still write full counts, but filters garbage years)
YEAR_MIN = 1900
YEAR_MAX = 2026


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def extract_year_from_goodreads_date(colname: str):
    # Goodreads date strings often end with " YYYY"
    return F.regexp_extract(F.col(colname), r"(\d{4})$", 1).cast("int")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default=DEFAULT_BOOKS_PARQUET)
    ap.add_argument("--interactions", default=DEFAULT_INTERACTIONS_PARQUET)
    ap.add_argument("--reviews", default=DEFAULT_REVIEWS_PARQUET)
    ap.add_argument("--out", default=DEFAULT_OUT_BASE)
    args = ap.parse_args()

    spark = SparkSession.builder.appName("ds_context_overview").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    out_base = args.out.rstrip("/")


    log("A) Interactions per year...")
    inter = (
        spark.read.parquet(args.interactions)
        .select("date_added")
        .where(F.col("date_added").isNotNull() & (F.length("date_added") > 0))
        .withColumn("year", extract_year_from_goodreads_date("date_added"))
        .where(F.col("year").isNotNull() & (F.col("year") >= YEAR_MIN) & (F.col("year") <= YEAR_MAX))
    )

    interactions_year = (
        inter.groupBy("year")
        .agg(F.count("*").alias("n_interactions"))
        .orderBy("year")
    )

    out_inter = f"{out_base}/interactions_per_year"
    interactions_year.coalesce(1).write.mode("overwrite").option("header", True).csv(out_inter)
    log(f"   -> wrote {out_inter}")
    interactions_year.orderBy(F.col("year").desc()).limit(10).show(truncate=False)


    log("B) Publications per year...")
    books = (
        spark.read.parquet(args.books)
        .select("publication_year")
        .where(F.col("publication_year").rlike("^[0-9]{4}$"))
        .withColumn("year", F.col("publication_year").cast("int"))
        .where(F.col("year").isNotNull() & (F.col("year") >= YEAR_MIN) & (F.col("year") <= YEAR_MAX))
    )

    publications_year = (
        books.groupBy("year")
        .agg(F.count("*").alias("n_published_books"))
        .orderBy("year")
    )

    out_pub = f"{out_base}/publications_per_year"
    publications_year.coalesce(1).write.mode("overwrite").option("header", True).csv(out_pub)
    log(f"   -> wrote {out_pub}")
    publications_year.orderBy(F.col("year").desc()).limit(10).show(truncate=False)


    log("C) Reviews per year (total / rating-only / with-text)...")
    # Reviews JSON is big; only read needed columns.
    # Based on your schema: reviews has date_added, rating, review_text.
    reviews = (
        spark.read.parquet(args.reviews)
        .select("date_added", "rating", "review_text")
        .where(F.col("date_added").isNotNull() & (F.length("date_added") > 0))
        .withColumn("year", extract_year_from_goodreads_date("date_added"))
        .where(F.col("year").isNotNull() & (F.col("year") >= YEAR_MIN) & (F.col("year") <= YEAR_MAX))
    )

    # Define "with text" robustly: non-null and has at least 1 non-whitespace char
    has_text = F.col("review_text").isNotNull() & (F.length(F.trim(F.col("review_text"))) > 0)
    rating_present = F.col("rating").isNotNull()

    # total rows per year (whether rating/text exists or not)
    total_reviews_year = reviews.groupBy("year").agg(F.count("*").alias("n_reviews_total"))

    # rows with rating present AND no review text (rating-only)
    rating_only_year = (
        reviews.where(rating_present & (~has_text))
        .groupBy("year").agg(F.count("*").alias("n_reviews_rating_only"))
    )

    # rows with review text (with-text), regardless of rating
    with_text_year = (
        reviews.where(has_text)
        .groupBy("year").agg(F.count("*").alias("n_reviews_with_text"))
    )

    reviews_year = (
        total_reviews_year
        .join(rating_only_year, on="year", how="left")
        .join(with_text_year, on="year", how="left")
        .fillna({"n_reviews_rating_only": 0, "n_reviews_with_text": 0})
        .orderBy("year")
    )

    out_rev = f"{out_base}/reviews_per_year"
    reviews_year.coalesce(1).write.mode("overwrite").option("header", True).csv(out_rev)
    log(f"   -> wrote {out_rev}")
    reviews_year.orderBy(F.col("year").desc()).limit(10).show(truncate=False)

    spark.stop()
    log("DONE. Context CSVs written.")


if __name__ == "__main__":
    main()

