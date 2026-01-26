#!/usr/bin/env python3
from datetime import datetime
from pyspark.sql import SparkSession, functions as F, types as T

# ---- Constants ----
PATH_GENRES = "/user/s3761576/goodreads/goodreads_book_genres_initial.parquet"
PATH_BOOKS  = "/user/s3761576/goodreads/goodreads_books.parquet"
PATH_INTERACTIONS = "/user/s3761576/goodreads/goodreads_interactions_dedup.parquet"

OUT_BASE = "./genre_out"          # local FS or HDFS depending on your environment

# For defining "romantasy"
THRESH = 10
RATIO = 0.2

PUB_YEAR_MIN = 1960
PUB_YEAR_MAX = 2019

FANTASY_KEY = "fantasy, paranormal"
ROMANCE_KEY = "romance"
ROMANTASY_KEY = "romantasy"       # synthetic genre name

# ---- Helpers ----
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def safe_int(col):
    return F.when(F.col(col).rlike("^[0-9]+$"), F.col(col).cast("int")).otherwise(F.lit(None).cast("int"))

def main():
    spark = SparkSession.builder.appName(f"rq3_all_genre_trends_thresh_{THRESH}").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # 1) Load genres and convert struct -> map so we can explode
    log("1) Loading genres and converting to map<str,int>...")
    genre_map_type = T.MapType(T.StringType(), T.IntegerType(), True)

    g = (
        spark.read.parquet(PATH_GENRES)
        .select("book_id", "genres")
        .withColumn("genres_map", F.from_json(F.to_json("genres"), genre_map_type))
        .select("book_id", "genres_map")
        .persist()
    )

    g.show(1)

    # 2) Explode to (book_id, genre, votes), keep only votes >= THRESH
    log("2) Exploding genres_map -> (book_id, genre, votes) with thresholding...")
    gv = (
        g.select("book_id", F.explode("genres_map").alias("genre", "votes"))
         .fillna({"votes": 0})
    )

    # normal genre membership at threshold
    book_genre = (
        gv.where(F.col("votes") >= THRESH)
          .select("book_id", "genre")
          .distinct()
    )

    book_genre.show(1)

    # 3) Build synthetic romantasy membership at the SAME threshold
    log("3) Creating synthetic genre: romantasy = (fantasy, paranormal) AND romance ...")
    # per book: do we have enough votes for both keys?
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
        .withColumn("genre", F.lit(ROMANTASY_KEY))
        .select("book_id", "genre")
    )

    # union: all genres + romantasy as another "genre"
    book_genre_all = (
        book_genre.unionByName(romantasy_books)
                  .persist()
    )

    # free g (map) later after we derived what we need
    g.unpersist()

    log(f"   -> book_genre_all rows (book_id, genre): {book_genre_all.count()}")

    # 4) Load books pub_year
    log("4) Loading books (book_id + pub_year)...")
    books = (
        spark.read.parquet(PATH_BOOKS)
        .select("book_id", "publication_year")
        .withColumn("pub_year", safe_int("publication_year"))
        .drop("publication_year")
        .where((F.col("pub_year") >= PUB_YEAR_MIN) & (F.col("pub_year") <= PUB_YEAR_MAX))
        .select("book_id", "pub_year")
        .persist()
    )

    # ---------------- SUPPLY: publication-year genre shares ----------------
    log("5) Computing SUPPLY trend: share of published books in each genre...")
    total_pub = books.groupBy("pub_year").agg(F.countDistinct("book_id").alias("n_books_total"))

    pub_genre_counts = (
        books.join(book_genre_all, on="book_id", how="inner")
             .groupBy("pub_year", "genre")
             .agg(F.countDistinct("book_id").alias("n_books_genre"))
    )

    pub_genre_shares = (
        pub_genre_counts.join(total_pub, on="pub_year", how="inner")
                        .withColumn("share_books_genre", F.col("n_books_genre") / F.col("n_books_total"))
                        .orderBy("pub_year", "genre")
    )

    out_pub = f"{OUT_BASE}/pub_year_genre_shares_thresh_{THRESH}"
    pub_genre_shares.coalesce(1).write.mode("overwrite").option("header", True).csv(out_pub)
    log(f"   -> DONE SUPPLY CSV: {out_pub}")

    # ---------------- DEMAND: interaction-year genre shares ----------------
    log("6) Computing DEMAND trend: share of user interactions in each genre...")
    inter = (
        spark.read.parquet(PATH_INTERACTIONS)
        .select("book_id", "date_added")
        .where(F.col("date_added").isNotNull() & (F.length(F.col("date_added")) > 0))
        .withColumn("added_year", F.regexp_extract("date_added", r"(\d{4})$", 1).cast("int"))
        .where(F.col("added_year").isNotNull())
        .select("book_id", "added_year")
        .persist()
    )

    total_int = inter.groupBy("added_year").agg(F.count("*").alias("n_interactions_total"))

    int_genre_counts = (
        inter.join(book_genre_all, on="book_id", how="inner")
             .groupBy("added_year", "genre")
             .agg(F.count("*").alias("n_interactions_genre"))
    )

    int_genre_shares = (
        int_genre_counts.join(total_int, on="added_year", how="inner")
                        .withColumn("share_interactions_genre",
                                    F.col("n_interactions_genre") / F.col("n_interactions_total"))
                        .orderBy("added_year", "genre")
    )

    out_int = f"{OUT_BASE}/interaction_year_genre_shares_thresh_{THRESH}"
    int_genre_shares.coalesce(1).write.mode("overwrite").option("header", True).csv(out_int)
    log(f"   -> DONE DEMAND CSV: {out_int}")

    # quick sanity: show romantasy in last year
    log("   -> Preview: romantasy demand in latest year:")
    max_year = int_genre_shares.agg(F.max("added_year")).collect()[0][0]
    (int_genre_shares.where((F.col("added_year") == max_year) & (F.col("genre") == ROMANTASY_KEY))
                    .show(truncate=False))

    # cleanup
    book_genre_all.unpersist()
    books.unpersist()
    inter.unpersist()
    spark.stop()
    log("DONE.")

if __name__ == "__main__":
    main()
