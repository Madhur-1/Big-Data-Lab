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

spark.sparkContext.setLogLevel('WARN')

df = spark.readStream.format('kafka').option('kafka.bootstrap.servers', '10.188.0.2:9092').option("startingOffsets", "earliest").option('subscribe', 'lab7').load()
# df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
print(df.isStreaming)

df.printSchema()

# model_path = 'gs://big-data-lab-madhurj/Ass7/RandomForestModel'
# model = RandomForestClassificationModel.load(model_path)
# print('Model Loaded....')

def formVector(s):
    s = s.value
    print(s)
    target = 0
    spiecies = s[5]
    if spiecies == 'Iris-versicolor':
        target = 0
    elif spiecies == 'Iris-virginica':
        target = 1
    else:
        target = 2
    vec = Vectors.dense(float(s[1]),
                                  float(s[2]),
                                  float(s[3]),
                                  float(s[4]))
    return vec

# def predict(x):
#     result= model.transform(x).head().prediction
#     return 1



query = df.writeStream.foreach(formVector).outputMode('update').format('console').start()

query.awaitTermination()

print('Data loaded...')

