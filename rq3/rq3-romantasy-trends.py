#!/usr/bin/env python3
from datetime import datetime
from pyspark.sql import SparkSession, functions as F, types as T

# ========= PATHS =========
PATH_GENRES = "/user/s3761576/goodreads/goodreads_book_genres_initial.parquet"
PATH_BOOKS  = "/user/s3761576/goodreads/goodreads_books.parquet"
PATH_INTERA = "/user/s3761576/goodreads/goodreads_interactions_dedup.parquet"

# output folder (small results only)
# Use "file:./romantasy_out" for local FS, or "hdfs:/user/.../romantasy_out" for HDFS
OUT_BASE = "./romantasy_out"

# romantasy threshold (try 1, 5, 10)
THRESH = 10

# year filters (to avoid junk publication years)
PUB_YEAR_MIN = 1960
PUB_YEAR_MAX = 2019

# ========= HELPERS =========
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def safe_int(col: str):
    # turns '', null, non-numeric into null
    return F.when(F.col(col).rlike("^[0-9]+$"), F.col(col).cast("int")).otherwise(F.lit(None).cast("int"))

def main():
    spark = (
        SparkSession.builder
        .appName(f"romantasy_trends_thresh_{THRESH}")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------
    # 1) Load genres (book_id, fantasy_votes, romance_votes)
    # ------------------------------------------------------------
    log("1) Loading genres (book_id + 2 genre vote columns)...")
    g_raw = spark.read.parquet(PATH_GENRES).select(
        "book_id",
        F.col("genres").getField("fantasy, paranormal").alias("fantasy_votes"),
        F.col("genres").getField("romance").alias("romance_votes"),
    )

    # fill nulls with 0 for vote counts
    g = g_raw.fillna({"fantasy_votes": 0, "romance_votes": 0})

    # romantasy label (co-occurrence with threshold)
    romantasy = (
        g.select(
            "book_id",
            "fantasy_votes",
            "romance_votes",
            ( (F.col("fantasy_votes") >= THRESH) & (F.col("romance_votes") >= THRESH) ).cast("int").alias("is_romantasy"),
            ( (F.col("fantasy_votes") >= THRESH) & (F.col("romance_votes") <  THRESH) ).cast("int").alias("is_fantasy_only"),
            ( (F.col("fantasy_votes") <  THRESH) & (F.col("romance_votes") >= THRESH) ).cast("int").alias("is_romance_only"),
        )
        .persist()
    )

    n_books_labeled = romantasy.count()
    log(f"   -> romantasy label table ready. books labeled: {n_books_labeled}")

    # ------------------------------------------------------------
    # 2) Catalog evolution: join with publication_year
    # ------------------------------------------------------------
    log("2) Loading books (book_id + publication_year) and computing publication-year trend...")
    b = (
        spark.read.parquet(PATH_BOOKS)
        .select("book_id", "publication_year")
        .withColumn("pub_year", safe_int("publication_year"))
        .drop("publication_year")
        .where((F.col("pub_year") >= PUB_YEAR_MIN) & (F.col("pub_year") <= PUB_YEAR_MAX))
    )

    # join: keep only books with genre info + valid pub_year
    bg = (
        b.join(romantasy.select("book_id", "is_romantasy", "is_fantasy_only", "is_romance_only"), on="book_id", how="inner")
        .select("book_id", "pub_year", "is_romantasy", "is_fantasy_only", "is_romance_only")
    )

    pub_year_trend = (
        bg.groupBy("pub_year")
        .agg(
            F.countDistinct("book_id").alias("n_books"),
            F.sum("is_romantasy").alias("n_romantasy"),
            F.sum("is_fantasy_only").alias("n_fantasy_only"),
            F.sum("is_romance_only").alias("n_romance_only"),
        )
        .withColumn("share_romantasy", F.col("n_romantasy") / F.col("n_books"))
        .orderBy("pub_year")
    )

    # write small output
    out_pub = f"{OUT_BASE}/pub_year_trend_thresh_{THRESH}"
    pub_year_trend.coalesce(1).write.mode("overwrite").option("header", True).csv(out_pub)
    log(f"   -> DONE publication-year trend. wrote CSV to: {out_pub}")

    # driver-small preview (safe: one row per year)
    log("   -> Publication-year trend preview (last 10 years in data):")
    pub_year_trend.orderBy(F.col("pub_year").desc()).limit(10).show(truncate=False)

    # ------------------------------------------------------------
    # 3) User adoption evolution: join with interactions by date_added year
    # ------------------------------------------------------------
    log("3) Loading interactions (book_id + date_added) and computing user adoption trend...")
    i = (
        spark.read.parquet(PATH_INTERA)
        .select("book_id", "date_added")
        .where(F.col("date_added").isNotNull() & (F.length(F.col("date_added")) > 0))
    )

    # Parse date_added like: "Tue Mar 03 08:19:07 -0800 2015"
    i = (
        i.withColumn("added_year", F.regexp_extract("date_added", r"(\d{4})$", 1).cast("int"))
         .where(F.col("added_year").isNotNull())
    )

    # Join with romantasy label
    ig = (
        i.join(romantasy.select("book_id", "is_romantasy"), on="book_id", how="inner")
        .select("added_year", "is_romantasy")
    )

    adoption_trend = (
        ig.groupBy("added_year")
        .agg(
            F.count("*").alias("n_interactions"),
            F.sum("is_romantasy").alias("n_romantasy_interactions"),
        )
        .withColumn("share_romantasy_interactions", F.col("n_romantasy_interactions") / F.col("n_interactions"))
        .orderBy("added_year")
    )

    out_adopt = f"{OUT_BASE}/interaction_year_trend_thresh_{THRESH}"
    adoption_trend.coalesce(1).write.mode("overwrite").option("header", True).csv(out_adopt)
    log(f"   -> DONE interaction-year trend. wrote CSV to: {out_adopt}")

    log("   -> Interaction-year trend preview (last 10 years in data):")
    adoption_trend.orderBy(F.col("added_year").desc()).limit(10).show(truncate=False)

    # ------------------------------------------------------------
    # 4) Cleanup
    # ------------------------------------------------------------
    romantasy.unpersist()
    spark.stop()
    log("DONE. All outputs written; previews printed.")

if __name__ == "__main__":
    main()

