from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import udf, col, from_json, struct, decode, avg
from pyspark.sql.types import LongType, IntegerType, StringType, StructType, FloatType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

topic = "test"
hs = "10.152.0.2:9092"
pipeline_path = "gs://ayushtosh23/Pipeline"
model_path = "gs://ayushtosh23/LR_Model"

spark = SparkSession.builder \
    .appName("DataStreamReader") \
    .getOrCreate()

spark.sparkContext.setLogLevel('error')

schema = StructType() \
  .add("business_id", StringType()) \
  .add("cool", IntegerType()) \
  .add("date", TimestampType()) \
  .add("funny", IntegerType()) \
  .add("review_id", StringType()) \
  .add("stars", FloatType()) \
  .add("text", StringType()) \
  .add("useful", IntegerType()) \
  .add("user_id", StringType())

df = spark \
    .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", hs) \
  .option("subscribe", topic) \
  .load()

init_time = 0.00
count = 0

df_1 = df.select(from_json(decode(col("value"), 'utf-8'), schema).alias("data")).select("data.*")


pipelineFit = PipelineModel.load(pipeline_path)
df_2 = pipelineFit.transform(df_1)
print("1")

mlModel = LogisticRegressionModel.load(model_path)
print("2")

predictions = mlModel.transform(df_2)
print(predictions)
predictions = predictions.drop("rawPrediction", "probability")
print("3")

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="stars", 
    predictionCol="prediction", 
    metricName="f1"
)

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="stars", 
    predictionCol="prediction", 
    metricName="accuracy"
)


def write_accuracy(df, epoch_id):
    global init_time, count
    time_elapsed = time.time() - init_time
    init_time = time.time()
        #print("Time Elapsed: ", time_elapsed)
    count = df.count()
        #print("Messages Recieved: ", count)
    latency = time_elapsed/count
        #print("Latency: ", time_elapsed/count)
    temp_df = spark.createDataFrame(
        [
            (evaluator_f1.evaluate(df), evaluator_acc.evaluate(df), time_elapsed, count, latency),
        ],
        [f"Batch {epoch_id} F1 Score", f"Batch {epoch_id} Accuracy", f"Batch {epoch_id} Time Elapsed", f"Batch {epoch_id} Count", f"Batch {epoch_id} Latency"] 
    )
    temp_df.write.format("console").save()

print("4")
'''
def time_latency(df, epoch_id):
    global init_time, count
    
    time_elapsed = time.time() - init_time
    print("Time Elapsed: ", time elapsed)
    count = df.count()
    print("Messages Recieved: ", count)
    print("Latency: ", time_elapsed/count)
    init_time = time.time()
    
    print("Accuracy: ", evaluator_acc.evaluate(df))
    print("F1 Score: ", evaluator_f1.evaluate(df))
    if(init_time == None):
        init_time = time.time()
    elif(time.time() - init_time > 9.9):
        time_elapsed = time.time() - init_time
        print("Time Elapsed: ", time_elapsed)
        count = df.count()
        print("Messages Recieved: ", count)
        print("Latency: ", time_elapsed/count)
        init_time = time.time()
'''    
query_acc = predictions \
    .writeStream \
    .foreachBatch(write_accuracy) \
    .start() \
    .awaitTermination()