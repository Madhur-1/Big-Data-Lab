from kafka import KafkaProducer
from time import sleep
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import json

topic = "test"
path_file = "yelp_test.json"
hs = "10.152.0.2"

spark = SparkSession.builder \
  .appName("DataStreamPublisher") \
  .getOrCreate()

schema = StructType() \
  .add("business_id", StringType()) \
  .add("cool", IntegerType()) \
  .add("date", TimestampType()) \
  .add("funny", IntegerType()) \
  .add("review_id", StringType()) \
  .add("stars", IntegerType()) \
  .add("text", StringType()) \
  .add("useful", IntegerType()) \
  .add("user_id", StringType())

df = spark.read.json(path_file, schema=schema, header=True)
df_to_json = df.toJSON().collect()

producer = KafkaProducer(bootstrap_servers=hs)

for i in df_to_json:
    x = i.encode('utf-8')
    producer.send(topic, x)
    sleep(1)
