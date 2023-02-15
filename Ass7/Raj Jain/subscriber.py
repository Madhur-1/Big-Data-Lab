from pyspark import SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as f
from itertools import chain
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("ML_predictor").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

ip_address = "10.168.0.2:9092" 
df = spark.readStream.format("kafka").option("kafka.bootstrap.servers",ip_address).option("subscribe","lab7").load()

#set header
split_cols = f.split(df.value,',')
df = df.withColumn('Sepal_length',split_cols.getItem(0))
df = df.withColumn('Sepal_width',split_cols.getItem(1))
df = df.withColumn('Petal_length',split_cols.getItem(2))
df = df.withColumn('Petal_width',split_cols.getItem(3))
df = df.withColumn('Class',split_cols.getItem(4))

for col in ['Sepal_length','Sepal_width','Petal_length','Petal_width']:
    df = df.withColumn(col,df[col].cast('float'))

df.createOrReplaceTempView('iris')

model = PipelineModel.load('gs://rajlab7/model')

df = df.withColumn('true_label', df['Class'])

predictions = model.transform(df)


result = predictions.select(["Class","label","prediction"])
result = result.withColumn("accuracy", f.when(f.col("label")==f.col("prediction"), 1).otherwise(0))

query = result.writeStream.format('console').start()

query.awaitTermination()