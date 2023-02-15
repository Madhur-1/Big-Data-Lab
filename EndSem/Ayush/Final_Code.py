#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from functools import reduce
from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, Word2Vec, IDF, Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.ml.feature import SQLTransformer, HashingTF
import pyspark
import sys
sc = SparkContext()
spark = SparkSession(sc)

#Combining all files
x = []
for i in range(48):
        filepath = "gs://ayushtosh23/yelp_train.json/part-000"+str("{0:02d}".format(i))+"-c8441126-a063-4a28-bf49-1ec0d5f0d0ce-c000.json"
        x.append(filepath)
dataframes = map(lambda r: spark.read.json(r), x)
df = reduce(lambda df1, df2: df1.unionAll(df2), dataframes)
print((df.count(), len(df.columns)))

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
countVectors = CountVectorizer(inputCol="filtered", outputCol="cV", vocabSize=10000, minDF=5)
CleanData = VectorAssembler(inputCols = ['cV', 'funny', 'useful'], outputCol = 'features')

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, CleanData])
pipelineFit = pipeline.fit(df)
b1 = "gs://ayushtosh23/Pipeline"
pipelineFit.save(b1)
dataset = pipelineFit.transform(df)
dataset.show(5)


(trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed = 100)
lr = LogisticRegression(featuresCol="features", labelCol = 'stars')
model = lr.fit(trainingData)

predictions_train = model.transform(trainingData)
predictions_test = model.transform(testData)

eval_acc = MulticlassClassificationEvaluator(labelCol="stars", predictionCol="prediction", metricName="accuracy")
accuracy_train = eval_acc.evaluate(predictions_train)
accuracy_test = eval_acc.evaluate(predictions_test)
print("Accuracy_train: %s\nAccuarcy_test: %s\n" %(accuracy_train, accuracy_test))

eval_f1 = MulticlassClassificationEvaluator(labelCol="stars", predictionCol="prediction", metricName="f1")
f1_train = eval_f1.evaluate(predictions_train)
f1_test = eval_f1.evaluate(predictions_test)
print("F1_train: %s\nF1_test: %s\n" %(f1_train, f1_test))

bucket = "gs://ayushtosh23/LR_Model"
model.save(bucket)

