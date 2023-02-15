from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

spark = SparkSession\
    .builder\
    .appName("Iris-Prediction")\
    .config("spark.driver.extraClassPath", "/home/ubuntu/jars/spark-sql-kafka-0-10_2.12-3.1.2.jar,/home/ubuntu/jars/commons-pool2-2.11.0.jar")\
    .getOrCreate()

df = spark.readStream.format('kafka').option('kafka.bootstrap.servers', '10.188.0.2:9092').option("startingOffsets", "earliest").option('subscribe', 'quickstart-events').load()
df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
print(df.isStreaming)

df.printSchema()

query = df.writeStream.outputMode('update').format('console').start()

# query.awaitTermination()

print('Data loaded...')

