from pyspark import SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, DoubleType, IntegerType
from itertools import chain
from pyspark.ml import PipelineModel
import pyspark
import pyspark.sql.functions as f
from pyspark.sql.functions import from_json, udf, split
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField
print(pyspark.__version__)
import json


sc = SparkContext()
spark = (SparkSession.builder.config("spark.jars.packages","com.johnsnowlabs:spark-nlp_2.12:3.0.3").getOrCreate())

ip_address = "10.128.15.232:9092" 
df = spark.readStream.format("kafka").option("kafka.bootstrap.servers",ip_address).option("startingOffsets", "earliest").option("subscribe","KafkaYelp").load()
df_val = df.value
df_val = df_val.cast('string')


schema = StructType([ \
	StructField("business_id", StringType(), True), \
	StructField("cool", IntegerType(), True), \
	StructField("date", StringType(), True), \
	StructField("funny", IntegerType(), True), \
	StructField("review_id", StringType(), True), \
	StructField("stars", DoubleType(), True), \
	StructField("text", StringType(), True), \
	StructField("useful", IntegerType(), True), \
	StructField("user_id", StringType(), True) \
	])

""" Message Parsing """
cols = ["business_id", "cool", "date", "funny", "review_id", "stars", "text", "useful","useful_id"]
col_type = {"business_id":StringType(), "cool":IntegerType(), "date": StringType(), "funny":IntegerType(), "review_id": StringType(), "stars": DoubleType(), "text": StringType(), "useful":IntegerType(), "user_id":StringType()}

split_cols = split(df.value, ',')
for i in range(6):
	df = df.withColumn(cols[i], split_cols.getItem(i))
df = df.withColumn(cols[6], split(df.value, '"text":').getItem(1))
df = df.withColumn(cols[7], split_cols.getItem(-2))
df = df.withColumn(cols[8], split(df.value, ':').getItem(-1))

split_cols = split(df.value, ':')
for col in cols[:6]:
	df = df.withColumn(col, split(df[col], ':').getItem(1))
df = df.withColumn(cols[6], split(df.value, ',"useful":').getItem(0))
df = df.withColumn(cols[7], split_cols.getItem(1))
df = df.withColumn(cols[8], split(df.value,'}').getItem(0))

df = df.withColumn("stars", df["stars"].cast('float'))

df.createOrReplaceTempView("yelp")
df.printSchema()
#df.select("text", "stars").show(2)

""" Predictions """
model = PipelineModel.load('gs://yelp-test/LR_TFIDF_model_complete/')

df = df.withColumn('true_label', df['stars'])

predictions = model.transform(df)

""" Evaluation """
evaluator = MulticlassClassificationEvaluator(predictionCol = "prediction", labelCol = "stars", metricName = "f1")

output_df = predictions.withColumn("correct", f.when((f.col('prediction')==1.0) & (f.col('stars')==1.0),1).when((f.col('prediction')==2.0) & (f.col('stars')==2.0),1).when((f.col('prediction')==3.0) & (f.col('stars')==3.0),1).when((f.col('prediction')==4.0) & (f.col('stars')==4.0),1).when((f.col('prediction')==5.0) & (f.col('stars')==5.0),1).otherwise(0))

df_acc = output_df.select(f.format_number(f.avg('correct')*100,2).alias('accuracy'))

output_df2 = output_df[["prediction","stars", "correct"]]
output_df2.createOrReplaceTempView('output')
print("Streaming Begins")
query = output_df2.writeStream.queryName("output").outputMode('update').format('console').start()
query2 = df_acc.writeStream.outputMode('complete').format('console').start()

query.awaitTermination()
query2.awaitTermination()

evaluator.evaluate(predictions)




