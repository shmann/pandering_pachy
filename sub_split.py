#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Subsample and split dataset

Usage:
    $ spark-submit sub_split.py filename_interactions.parquet subsample_percent
    
If spark.yarn.executor.memoryOverhead error, try spark-submit --executor-memory 5G ...

E.g.

spark-submit sub_split.py data/parquets/goodreads_interactions_poetry.parquet 1

spark-submit --executor-memory 6G sub_split.py goodreads_interactions.parquet 100

'''

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand,col

def main(spark, filename_interactions, pct):

    min_interactions=10
    subsample_seed=1
    split_seed_622=1
    split_seed_55=1
    vshuff_seed=1
    tshuff_seed=1

    print("reading {}".format(filename_interactions))
    interactions=spark.read.parquet(filename_interactions)

    #eliminate users with fewer than min_interactions interactions
    interactions.createOrReplaceTempView("interactions")
    interactions=spark.sql(" \
        SELECT * \
        FROM interactions \
        WHERE user_id IN \
            ( \
                SELECT user_id \
                FROM interactions \
                GROUP BY user_id \
                HAVING COUNT(*) >={} \
            ) \
        ORDER BY user_id \
    ".format(min_interactions))

    #get random subsample if not 100%
    if pct=="100":
        users=interactions.select("user_id").distinct()
    else:
        users=interactions.select("user_id").distinct().sample(False,int(pct)/100.,seed=subsample_seed).cache()

    #split users randomly
    users_train,users_val,users_test=users.randomSplit(weights=[.6,.2,.2],seed=split_seed_622)

    #split interactions
    train=interactions.join(users_train,on="user_id",how="inner")
    val=interactions.join(users_val,on="user_id",how="inner")
    test=interactions.join(users_test,on="user_id",how="inner")

    #shuffle val and test
    val=val.withColumn("rand",rand(seed=vshuff_seed)).sort("user_id","rand").drop("rand")
    test=test.withColumn("rand",rand(seed=tshuff_seed)).sort("user_id","rand").drop("rand")

    print("adding index to val and test...")
    val=val.rdd.zipWithIndex().map(lambda row: (row[1],)+tuple(row[0])).toDF(["index"]+val.columns)
    test=test.rdd.zipWithIndex().map(lambda row: (row[1],)+tuple(row[0])).toDF(["index"]+test.columns)

    #split interactions from val and test 50:50 
    val_train=val.filter(val.index%2==0).drop("index")
    val=val.filter(val.index%2==1).drop("index")
    test_train=test.filter(test.index%2==0).drop("index")
    test=test.filter(test.index%2==1).drop("index")

    #merge half of val and test into train
    train=train.union(val_train).union(test_train)

    #remove books with no training data
    trained_books=train.select("book_id").distinct()
    val=val.join(trained_books,on="book_id",how="inner")
    test=test.join(trained_books,on="book_id",how="inner")

    print("writing {}...".format("val_pct_"+pct+".parquet"))
    val.write.parquet("val_pct_"+pct+".parquet")
    #val.show()
    print("writing {}...".format("test_pct_"+pct+".parquet"))
    test.write.parquet("test_pct_"+pct+".parquet")
    #test.show()
    print("writing {}...".format("train_pct_"+pct+".parquet"))
    train.write.parquet("train_pct_"+pct+".parquet")
    #train.show()
    print("done")

if __name__=="__main__":

    spark=SparkSession.builder.appName("sub_split").getOrCreate()

    filename_interactions=sys.argv[1]
    pct=sys.argv[2]

    main(spark, filename_interactions, pct)

