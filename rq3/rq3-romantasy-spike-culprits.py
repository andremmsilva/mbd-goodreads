#!/usr/bin/env python3
from datetime import datetime
from pyspark.sql import SparkSession, functions as F, types as T

# -------- PARAMETERS --------
THRESH = 10
RATIO = 0.2

START_YEAR = 2001
END_YEAR = 2004

FANTASY_KEY = "fantasy, paranormal"
ROMANCE_KEY = "romance"

# -------- PATHS --------
PATH_GENRES = "/user/s3761576/goodreads/goodreads_book_genres_initial.parquet"
PATH_BOOKS  = "/user/s3761576/goodreads/goodreads_books.parquet"
PATH_INTERACTIONS = "/user/s3761576/goodreads/goodreads_interactions_dedup.parquet"

OUT_PATH = f"./romantasy_out/top_romantasy_books_{START_YEAR}_{END_YEAR}_THRESH_{THRESH}"

# -------- Helpers --------
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# -------- Main --------
def main():
    spark = SparkSession.builder.appName("romantasy_spike_culprits").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # 1) Load genres and compute strict romantasy membership
    log("1) Computing strict romantasy book set...")
    genre_map_type = T.MapType(T.StringType(), T.IntegerType(), True)

    g = (
        spark.read.parquet(PATH_GENRES)
        .select("book_id", "genres")
        .withColumn("genres_map", F.from_json(F.to_json("genres"), genre_map_type))
        .select("book_id", "genres_map")
    )

    per_book_votes = (
        g.select(
            "book_id",
            F.coalesce(F.col("genres_map").getItem(FANTASY_KEY), F.lit(0)).alias("fantasy_votes"),
            F.coalesce(F.col("genres_map").getItem(ROMANCE_KEY), F.lit(0)).alias("romance_votes"),
        )
    )

    romantasy_books = (
        per_book_votes
        .where(
            (F.col("fantasy_votes") >= THRESH) &
            (F.col("romance_votes") >= THRESH) &
            (
                F.least(F.col("fantasy_votes"), F.col("romance_votes")) /
                F.greatest(F.col("fantasy_votes"), F.col("romance_votes"))
                >= F.lit(RATIO)
            )
        )
        .select("book_id")
        .distinct()
        .persist()
    )

    log(f"   -> romantasy books identified: {romantasy_books.count()}")

    # 2) Load interactions and filter to spike years
    log("2) Filtering interactions to spike window...")
    inter = (
        spark.read.parquet(PATH_INTERACTIONS)
        .select("book_id", "date_added")
        .where(F.col("date_added").isNotNull())
        .withColumn("added_year", F.regexp_extract("date_added", r"(\d{4})$", 1).cast("int"))
        .where((F.col("added_year") >= START_YEAR) & (F.col("added_year") <= END_YEAR))
        .select("book_id")
    )

    # 3) Count interactions per romantasy book
    log("3) Counting interactions per romantasy book...")
    romantasy_inter = inter.join(romantasy_books, on="book_id", how="inner")

    total_all = inter.count()
    total_rom = romantasy_inter.count()
    log(f"Total interactions in window: {total_all:,}")
    log(f"Romantasy interactions in window: {total_rom:,}")

    book_counts = (
        romantasy_inter.groupBy("book_id")
        .agg(F.count("*").alias("n_romantasy_interactions"))
    )

    books = spark.read.parquet(PATH_BOOKS).select("book_id", "title")

    result = (
        book_counts.join(books, on="book_id", how="left")
        .withColumn("share_within_romantasy", F.col("n_romantasy_interactions") / F.lit(total_rom))
        .withColumn("share_of_total", F.col("n_romantasy_interactions") / F.lit(total_all))
        .orderBy(F.col("share_within_romantasy").desc(), F.col("n_romantasy_interactions").desc())
    )

    # 4) Write top books to CSV
    log("5) Writing top romantasy books table...")
    result.limit(50).coalesce(1).write.mode("overwrite").option("header", True).csv(OUT_PATH)

    log(f"DONE. Output written to {OUT_PATH}")
    spark.stop()

if __name__ == "__main__":
    main()
