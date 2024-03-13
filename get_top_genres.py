#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Get top genres for books

Usage:
    $ spark-submit get_top_genres.py filename_genres.json filename_book_id_map.csv output_filename_top_genres.parquet
    
If spark.yarn.executor.memoryOverhead error, try spark-submit --executor-memory 5G ...

E.g.

spark-submit get_top_genres.py data/poetry/goodreads_book_genres_initial.json data/poetry/book_id_map_poetry.csv top_genres_poetry.parquet

spark-submit get_top_genres.py goodreads_book_genres_initial.json hdfs:/user/bm106/pub/goodreads/book_id_map.csv goodreads_top_genres.parquet

'''

import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import Row

def main(spark, filename_genres, filename_book_map, filename_top_genres):

    print("reading {}".format(filename_genres))

    genres = spark.read.json(filename_genres)
    genres = genres.select("book_id", F.col("genres.*")).na.fill(0)

    def argmax(cols, *args):
        return [c for c, v in zip(cols, args) if v == max(args)][0]

    argmax_udf = lambda cols: F.udf(lambda *args: argmax(cols, *args), StringType())

    genre_columns = genres.columns
    genre_columns.remove("book_id")

    genres=genres.withColumn("top_genre", argmax_udf(genre_columns)(*genre_columns)).select("book_id", "top_genre")

    distinct_genres = genres.select("top_genre").distinct().sort("top_genre")

    genre_map_schema = StructType(distinct_genres.schema.fields[:] + [StructField("genre_id", LongType(), False)])

    print("creating genre map...")
    
    distinct_genres_indexed_rdd = distinct_genres.rdd.zipWithIndex()
    
    genre = (distinct_genres_indexed_rdd.map(lambda ri: row_with_index(*list(ri[0]) + [ri[1]])).toDF(genre_map_schema))

    genre_map = (distinct_genres_indexed_rdd.map(lambda ri: Row(*list(ri[0]) + [ri[1]])).toDF(genre_map_schema)) 

    genres = genres.join(genre_map, on="top_genre", how="inner")
    
    print("reading {}".format(filename_book_map))
    
    book_map = spark.read.csv(filename_book_map, header=True, schema="book_id_csv INT, book_id INT")

    genres = genres.join(book_map, on="book_id", how="inner").selectExpr("book_id_csv as book_id","genre_id","top_genre").sort("book_id")

    print("writing {}".format(filename_top_genres))

    genres.write.parquet(filename_top_genres)

if __name__=="__main__":

    spark=SparkSession.builder.appName("get_top_genres").getOrCreate()

    filename_genres=sys.argv[1]
    filename_book_id_map=sys.argv[2]
    filename_top_genres=sys.argv[3]

    main(spark, filename_genres, filename_book_id_map, filename_top_genres)

