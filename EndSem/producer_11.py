import pyspark
import json
from time import sleep
from kafka import KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql.types import *


# brokers = 'localhost:9092'
# The IP can be found from Compute Engine in GCP. It is the internal IP associated with kafka.
brokers = '10.188.0.2:9092'
topic = 'project'
data_path = "gs://bdl2022/YELP_train.csv"

# api_version is the version of kafka-python installed.
producer = KafkaProducer(bootstrap_servers=[brokers],
                         api_version = (2,0,2),
                         value_serializer=lambda x: 
                         x.encode('utf-8') 
                         )

spark = SparkSession.builder.appName("YELP").getOrCreate()

schema = StructType([
    StructField("business_id", StringType()),
    StructField("cool", IntegerType()),
    StructField("date", StringType()),
    StructField("funny", IntegerType()),
    StructField("review_id", StringType()),
    StructField("stars", FloatType()),
    StructField("text", StringType()),
    StructField("useful", IntegerType()),
    StructField("user_id", StringType()),
])
df = spark.read.json(data_path, schema=schema, header=True)
df_to_json = df.toJSON().collect()

for i in df_to_json:
    x = i.encode('utf-8')
    producer.send(topic, x)
    sleep(1)