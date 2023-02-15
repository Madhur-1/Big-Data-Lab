import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 pyspark-shell'
from pyspark.sql.types import StructType, StringType, FloatType
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
import pyspark.sql.functions as f
from pyspark.ml import PipelineModel
from itertools import chain

spark = SparkSession\
    .builder\
    .appName("Iris-Prediction")\
    .config("spark.driver.extraClassPath", "/home/ubuntu/jars/spark-sql-kafka-0-10_2.12-3.1.2.jar,/home/ubuntu/jars/commons-pool2-2.11.0.jar")\
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')

df = spark.readStream.format('kafka').option('kafka.bootstrap.servers', '10.188.0.2:9092').option("startingOffsets", "earliest").option('subscribe', 'json-iris-data').option("failOnDataLoss", "false").load()
df = df.selectExpr("CAST(value AS STRING)")

schema = StructType()\
    .add("sepal_length", FloatType())\
    .add("sepal_width", FloatType())\
    .add("petal_length", FloatType())\
    .add("petal_width", FloatType())\
    .add("class", StringType())

print(df.isStreaming)

df.printSchema()

df = df.select(f.from_json(f.decode(df.value, 'utf-8'), schema=schema).alias("input"))
df = df.select("input.*")

query3 = df.writeStream.outputMode('update').format('console').start()

model_path = 'gs://big-data-lab-madhurj/Ass7/Pipeline_Model'
model = PipelineModel.load(model_path)
print('Model Loaded....')

predictions = model.transform(df)

mapping = dict(zip([0.0,1.0,2.0], ['Iris-setosa','Iris-versicolor','Iris-virginica']))
mapping_expr = f.create_map([f.lit(x) for x in chain(*mapping.items())])
output_df = predictions.withColumn('prediction', mapping_expr[f.col("prediction")])[['prediction','species']]

output_df = output_df.withColumn('correct', f.when((f.col('prediction')=='Iris-setosa') & (f.col('species')=='Iris-setosa'),1).when((f.col('prediction')=='Iris-versicolor') & (f.col('species')=='Iris-versicolor'),1).when((f.col('prediction')=='Iris-virginica') & (f.col('species')=='Iris-virginica'),1).otherwise(0))

df_acc = output_df.select(f.format_number(f.avg('correct')*100,2).alias('accuracy'))

output_df2 = output_df[['prediction','species','correct']]
output_df2.createOrReplaceTempView('output')

query1 = output_df2.writeStream.queryName("output").outputMode('update').format('console').start()
query2 = df_acc.writeStream.outputMode('update').format('console').start()

query1.awaitTermination()
query2.awaitTermination()

