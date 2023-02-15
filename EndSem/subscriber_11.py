import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import split,decode,substring
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, IndexToString
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 pyspark-shell'

spark = SparkSession.builder.appName("iris").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# brokers = 'localhost:9092'
brokers = '10.128.15.227:9092'
topic = 'project'
model_path = "gs://me18b029/model" # Change Model Path

df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", brokers) \
  .option("subscribe", topic) \
  .load()

# Change schema
schema = StructType() \
  .add("sepal_length", FloatType()) \
  .add("sepal_width", FloatType()) \
  .add("petal_length", FloatType()) \
  .add("petal_width", FloatType()) \
  .add("species", StringType())

df = df.select(f.from_json(f.decode(df.value, 'utf-8'), schema=schema).alias("input"))
df = df.select("input.*")

from pyspark.ml import PipelineModel
model = PipelineModel.load(model_path)
pred_df = model.transform(df)
# print(pred_df)
pred_df = pred_df.select(pred_df.prediction, pred_df.labelIndex)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator_acc = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='labelIndex',metricName='accuracy')
evaluator_f1 = MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='labelIndex',metricName='f1')

def func(df, epoch):
  from pyspark.sql import Row

  acc = evaluator_acc.evaluate(df)*100 
  acc_row = Row(ba=acc)
  acc_df = spark.createDataFrame([acc_row])
  acc_col = f"Batch {epoch} Accuracy"
  acc_df = acc_df.withColumnRenamed('ba',acc_col) 

  f1 = evaluator_f1.evaluate(df)*100 
  f1_row = Row(bf=f1)
  f1_df = spark.createDataFrame([f1_row])
  f1_col = f"Batch {epoch} F1 Score"
  f1_df = f1_df.withColumnRenamed('bf',f1_col) 

  pred_lab_df = df.select(df.prediction, df.labelIndex)
  col1_name = f"Batch {epoch} Predicted Review" 
  col2_name = f"Batch {epoch} True Review" 

  pred_lab_df = pred_lab_df.withColumnRenamed('pred_review',col1_name)
  pred_lab_df = pred_lab_df.withColumnRenamed('review',col2_name)

  pred_lab_df.write.format("console").save()
  acc_df.write.format("console").save()

query = pred_df.writeStream \
        .option("truncate",False) \
        .foreachBatch(func) \
        .start()

query.awaitTermination()

# Jar files to be used while deploying

# gs://bdl2022/lab7/jar_files/commons-pool2-2.6.2.jar
# gs://bdl2022/lab7/jar_files/kafka-clients-2.6.0.jar
# gs://bdl2022/lab7/jar_files/lz4-java-1.7.1.jar
# gs://bdl2022/lab7/jar_files/scala-library-2.12.10.jar
# gs://bdl2022/lab7/jar_files/slf4j-api-1.7.30.jar
# gs://bdl2022/lab7/jar_files/snappy-java-1.1.7.3.jar
# gs://bdl2022/lab7/jar_files/spark-sql-kafka-0-10_2.12-3.1.1.jar
# gs://bdl2022/lab7/jar_files/spark-tags_2.12-3.1.1.jar
# gs://bdl2022/lab7/jar_files/spark-token-provider-kafka-0-10_2.12-3.0.0-preview2.jar
# gs://bdl2022/lab7/jar_files/unused-1.0.0.jar
# gs://bdl2022/lab7/jar_files/zstd-jni-1.4.4-7.jar