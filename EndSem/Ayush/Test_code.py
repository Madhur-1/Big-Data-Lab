#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
sc = SparkContext()
spark = SparkSession(sc)

filepath = "gs://ayushtosh23/yelp_train.json/part-00000-c8441126-a063-4a28-bf49-1ec0d5f0d0ce-c000.json"
df = spark.read.json(filepath)
print((df.count(), len(df.columns)))
df.printSchema()
df.show()

pipelineFit = PipelineModel.load('gs://ayushtosh23/Pipeline')
dataset = pipelineFit.transform(df)

model = LogisticRegressionModel.load('gs://ayushtosh23/LR_Model')
predictions = model.transform(dataset)

eval_acc = MulticlassClassificationEvaluator(labelCol="stars", predictionCol="prediction", metricName="accuracy")
accuracy = eval_acc.evaluate(predictions)
print("Accuracy: %s\n" %(accuracy))

eval_f1 = MulticlassClassificationEvaluator(labelCol="stars", predictionCol="prediction", metricName="f1")
f1 = eval_f1.evaluate(predictions)
print("F1_train: %s\n" %(f1))


# In[ ]:




