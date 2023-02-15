from __future__ import print_function 
from pyspark.context import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
import numpy as np
from pyspark.ml.feature import StandardScaler
import pyspark.sql.functions as f
import pyspark.sql.types
from pyspark.sql import Row
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import PCA 

sc = SparkContext()
spark = SparkSession(sc)

## Reading Dataframe 

spark_df = spark.read.format("bigquery").option("table", "rare-guide-340916.iris_dataset.iris_2022").load().toDF("sl", "sw", "pl", "pw", "labclass")
clean_data = spark_df.withColumn("label", spark_df["labclass"])

cols = spark_df.drop('labclass').columns

assembler = VectorAssembler(inputCols=cols, outputCol = 'features')
labelIndexer = StringIndexer(inputCol="labclass", outputCol="indexedLabel").fit(spark_df)

## Standardize the columns

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=False, withMean=True)

## Principal component analysis

pca = PCA(k=3, inputCol='scaledFeatures', outputCol='pcaFeature')

(trainingData, testData) = spark_df.randomSplit([0.8, 0.2])

## Training a RandomForest model

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="pcaFeature", numTrees=10)

## Retrieve orginal labels from indexed labels

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

## Modying indexers and forest in a Pipeline

pipeline = Pipeline(stages=[labelIndexer, assembler,scaler,pca, rf, labelConverter])

## Train the ML model

model = pipeline.fit(trainingData)

## Predictions

predictions = model.transform(testData)
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set error fraction = %g" % (1.0 - accuracy))
