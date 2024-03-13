#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Train and test ALS model

Usage:
    $ spark-submit als_rec.py filename_train.parquet filename_test.parquet output_filename.parquet
    
If spark.yarn.executor.memoryOverhead error, try spark-submit --executor-memory 6G ...

E.g.

spark-submit --driver-memory 16G als_rec_correct.py data/parquets/train_pct_100.parquet data/parquets/val_pct_100.parquet pct_100_als_model.parquet

spark-submit --executor-memory 8G als_rec.py good/train_pct_100.parquet good/val_pct_100.parquet pct_100_als_model.parquet

'''

from pyspark.sql import SparkSession
import sys
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics

def main(spark, filename_train, filename_test, filename_output):
    
    #ranks=[50]
    #regParams=[.05]
    #ranks=[9,3,6,1,12,15,18,21]
    #regParams=[.001,.002,.005,.01,.02,.05,.1,.2,.5,1]
    #ranks=[9,3,6]
    #regParams=[.01,.02,.05,.1]
    ranks=[10,20,40,80,160]
    regParams=[.001,.002,.005,.01,.02,.05,.1,.2,.5,1]
    
    als_seed=1
    k=500
    
    print("reading {}".format(filename_train))
    train = spark.read.parquet(filename_train)
    print("reading {}".format(filename_test))
    test = spark.read.parquet(filename_test)
    
    best_rank = None
    best_regParam = None
    best_mapAtk = None
    best_model = None
    
    r_rmse=[]
    r_test_mapAtk=[]
    r_test_precisionAtk=[]
    r_test_ndcgAtk=[]
    r_mapAtk=[]
    r_precisionAtk=[]
    r_ndcgAtk=[]
    
    for rank in ranks:
        for regParam in regParams:
            als = ALS(
                rank=rank # default 10
                , maxIter=10
                , regParam=regParam # default .1
                , numUserBlocks=10
                , numItemBlocks=10
                , implicitPrefs=False
                , alpha=1.0
                , userCol='user_id'
                , itemCol='book_id'
                , seed=als_seed
                , ratingCol='rating'
                , nonnegative=False
                , checkpointInterval=10
                , intermediateStorageLevel='MEMORY_AND_DISK'
                , finalStorageLevel='MEMORY_AND_DISK'
                , coldStartStrategy='drop' # default 'nan'
            )
            
            print("fitting model with rank={} and regParam={}...".format(rank,regParam))
            model = als.fit(train)
            
            #for use when reloading previous models
            #model = ALSModel.load("model_test")
            
            if len(ranks)==1 and len(regParams)==1:
                print("writing book_factors_ and model_{}".format(filename_output))
                model.itemFactors.write.parquet("book_factors_"+filename_output)
                model.save("model_"+filename_output)
                pass
            
            predictions = model.transform(test)
            
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
            
            print("evaluating RMSE for all predictions in the test set...")
            rmse = evaluator.evaluate(predictions)
            print("RMSE for transformed test set: {}".format(rmse))
            r_rmse.append(rmse)
            
            #only keep books where rating is 3+ (ranking metrics do not take into account order of labels)
            Labels=test.where("rating>=3").groupBy("user_id").agg(F.expr("collect_list(book_id) as book_ids"))
            
            ###OPTION 1: ONLY USE TRANSFORMED TEST SET
            
            ###top_k_test_window = Window.partitionBy(predictions["user_id"]).orderBy(predictions["prediction"].desc())
            ###top_k_test_predictions = predictions.select("user_id", "book_id", "rating", "prediction", F.rank().over(top_k_test_window).alias("rank")).where("rank <= {0}".format(k))
            
            #get top k predictions specifically for interactions in the test set
            ###test_window = Window.partitionBy(top_k_test_predictions["user_id"]).orderBy(top_k_test_predictions["prediction"].desc())
            ###test_prediction = top_k_test_predictions.withColumn("book_ids", F.collect_list("book_id").over(test_window))
            ###test_prediction = test_prediction.groupBy("user_id").agg(F.max("book_ids").alias("book_ids"))
            
            #build predictionAndLabels for top k predictions in the test set 
            ###test_predictionAndLabels = test_prediction.join(Labels, "user_id").rdd.map(lambda row: (row[1], row[2]))
            
            #print("evaluating ranking metrics for the top {} predictions in the test set...".format(k))
            ###test_metrics = RankingMetrics(test_predictionAndLabels)
            ###test_mapAtk = test_metrics.meanAveragePrecision
            ###test_precisionAtk = test_metrics.precisionAt(k)
            ###test_ndcgAtk = test_metrics.ndcgAt(k)
            
            ###r_test_mapAtk.append(test_mapAtk)
            ###r_test_precisionAtk.append(test_precisionAtk)
            ###r_test_ndcgAtk.append(test_ndcgAtk)
            
            #note: k forced at previous step, thus all metrics are at k, not just *At(k) metrics
            ###print("MAP at {}: {}".format(k, test_mapAtk))
            ###print("Precision at {}: {}".format(k, test_precisionAtk))
            ###print("NDCG at {}: {}".format(k, test_ndcgAtk))
            
            ###END OPTION 1
            
            #OPTION 2: GET ALL k PREDICTIONS
            prediction = model.recommendForUserSubset(test.select("user_id").distinct(), k)
            
            #OPTION 2.1: USE ALL k TOP PREDICTIONS
            prediction = prediction.select("user_id","recommendations.book_id")
            
            #OPTION 2.2: USE SUBSET OF k TOP PREDICTIONS
            ####prediction = prediction.select("user_id",F.expr("explode(recommendations)")).select("user_id","col.book_id","col.rating")
            
            #OPTION 2.2.1: REMOVE PREDICTIONS ASSOCIATED WITH INTERACTIONS IN TRAINING SET
            ####prediction = prediction.join(train.select("user_id","book_id"), ["user_id", "book_id"], "leftanti")
            
            #OPTION 2.2.2: KEEP ONLY PREDICTIONS ASSOCIATED WITH INTERACTIONS IN THE TEST SET
            ####prediction = prediction.join(test.select("user_id","book_id"), ["user_id", "book_id"], "inner")
            
            #REQUIRED FOR SUBSETS (OPTION 2.2)
            #aggregate for ranking metrics; ensure correct order
            ####prediction_window = Window.partitionBy("user_id").orderBy(F.desc("rating"))
            ####prediction = prediction.withColumn("book_ids", F.collect_list("book_id").over(prediction_window))
            ####prediction = prediction.groupBy("user_id").agg(F.max("book_ids").alias("book_ids"))
            
            #build predictionAndLabels for top k predictions
            predictionAndLabels = prediction.join(Labels, "user_id").rdd.map(lambda row: (row[1], row[2]))
            
            print("evaluating ranking metrics for the top {} predictions...".format(k))
            metrics = RankingMetrics(predictionAndLabels)
            mapAtk = metrics.meanAveragePrecision
            precisionAtk = metrics.precisionAt(k)
            ndcgAtk = metrics.ndcgAt(k)
            
            r_mapAtk.append(mapAtk)
            r_precisionAtk.append(precisionAtk)
            r_ndcgAtk.append(ndcgAtk)
            
            #note: k forced at previous step, thus all metrics are at k, not just *At(k) metrics
            print("MAP at {}: {}".format(k, mapAtk))
            print("Precision at {}: {}".format(k, precisionAtk))
            print("NDCG at {}: {}".format(k, ndcgAtk))
            
            if best_mapAtk == None or mapAtk > best_mapAtk:
                best_rank = rank
                best_regParam = regParam
                best_mapAtk = mapAtk
                best_model = model
    
    if best_mapAtk != None:
        print("Best MAP at {}: {}".format(k, best_mapAtk))
    
    book_factors = best_model.itemFactors
    
    if len(ranks)!=1 and len(regParams)!=1:
        print("writing book_factors_ and model_{}".format(filename_output))
        book_factors.write.parquet("book_factors_"+filename_output)
        best_model.save("model_"+filename_output)
    
    print(r_rmse)
    print(r_test_mapAtk)
    print(r_test_precisionAtk)
    print(r_test_ndcgAtk)
    print(r_mapAtk)
    print(r_precisionAtk)
    print(r_ndcgAtk)
    
    print("done")
    
if __name__=="__main__":
    
    spark=SparkSession.builder.appName("als_rec").getOrCreate()
    
    filename_train=sys.argv[1]
    filename_test=sys.argv[2]
    filename_output=sys.argv[3]
    
    main(spark, filename_train, filename_test, filename_output)

