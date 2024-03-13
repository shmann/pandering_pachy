#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Merge top genres with book factors

Usage:
    $ spark-submit merge_top_genres.py filename_book_factors.parquet filename_top_genres.parquet output_filename_book_factors_top_genres_output.csv
    
If spark.yarn.executor.memoryOverhead error, try spark-submit --executor-memory 5G ...

E.g.

spark-submit merge_top_genres.py data/parquets/pct_25_als_model_book_factors_poetry.parquet data/parquets/top_genres_poetry.parquet pct_25_als_model_book_factors_top_genres_poetry.csv

spark-submit merge_top_genres.py pct_100_als_model_book_factors.parquet goodreads_top_genres.parquet pct_100_als_model_book_factors_top_genres.csv

'''

import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main(spark, filename_book_factors, filename_top_genres, filename_book_factors_top_genres):

    print("reading {}".format(filename_book_factors))
    book_factors = spark.read.parquet(filename_book_factors)
    
    print("reading {}".format(filename_top_genres))
    top_genres = spark.read.parquet(filename_top_genres)

    book_factors_top_genres = book_factors.join(top_genres, book_factors.id==top_genres.book_id, how="inner")

    n_features = book_factors_top_genres.select(F.size("features")).take(1)[0][0]
    
    book_factors_top_genres = book_factors_top_genres.select(["book_id", "genre_id"]+[F.expr("features[" + str(x) + "] as x" + str(x)) for x in range(0, n_features)])
    
    print("writing {}".format(filename_book_factors_top_genres))
    book_factors_top_genres.write.csv(filename_book_factors_top_genres)

if __name__=="__main__":

    spark=SparkSession.builder.appName("merge_top_genres").getOrCreate()

    filename_book_factors=sys.argv[1]
    filename_top_genres=sys.argv[2]
    filename_book_factors_top_genres=sys.argv[3]

    main(spark, filename_book_factors, filename_top_genres, filename_book_factors_top_genres)

