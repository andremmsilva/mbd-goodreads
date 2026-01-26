#!/usr/bin/env python3
import os
import sys
import traceback
from pprint import pformat
from datetime import datetime

from pyspark.sql import SparkSession

# -------- CONFIG --------
OUT_DIR = "./schemas"

DATASETS = [
    # name, format, path
    ("goodreads_book_authors", "parquet",     "/user/s3761576/goodreads/goodreads_book_authors.parquet"),
    ("goodreads_books",        "parquet",  "/user/s3761576/goodreads/goodreads_books.parquet"),
    ("goodreads_interactions", "parquet",  "/user/s3761576/goodreads/goodreads_interactions_dedup.parquet"),
    ("goodreads_reviews",      "parquet",     "/user/s3761576/goodreads/goodreads_reviews_dedup.parquet"),
    ("goodreads_genres",      "parquet",     "/user/s3761576/goodreads/goodreads_book_genres_initial.parquet"),
]
# ------------------------


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)  # atomic rename


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("goodreads_dump_all_schemas")
        .getOrCreate()
    )

    log(f"Writing schemas to local directory: {os.path.abspath(OUT_DIR)}")
    log(f"Datasets queued: {len(DATASETS)}")

    ok = 0
    fail = 0

    for i, (name, fmt, path) in enumerate(DATASETS, start=1):
        out_path = os.path.join(OUT_DIR, f"{name}.schema.txt")
        log(f"[{i}/{len(DATASETS)}] START {name} ({fmt}) :: {path}")

        try:
            if fmt == "json":
                df = spark.read.json(path)
            elif fmt == "parquet":
                df = spark.read.parquet(path)
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            schema_str = df._jdf.schema().treeString()
            sample_str = pformat(df.take(5))

            content = []
            content.append(f"# name: {name}")
            content.append(f"# format: {fmt}")
            content.append(f"# input: {path}")
            content.append("")
            content.append(schema_str)
            content.append("")
            content.append(sample_str)
            content.append("")

            write_text(out_path, "\n".join(content))

            ok += 1
            log(f"[{i}/{len(DATASETS)}] DONE  {name} -> saved schema at {out_path}")

        except Exception as e:
            fail += 1
            log(f"[{i}/{len(DATASETS)}] FAIL  {name}: {e}")
            tb = traceback.format_exc()
            # write the error to a file too, so you don't lose it in console spam
            err_path = os.path.join(OUT_DIR, f"{name}.ERROR.txt")
            write_text(err_path, tb)
            log(f"[{i}/{len(DATASETS)}]      error details saved at {err_path}")
            # continue to next dataset

    log(f"Finished. Success={ok}, Failed={fail}")
    spark.stop()

    # exit nonzero if anything failed (handy for CI / quick checking)
    if fail > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()

