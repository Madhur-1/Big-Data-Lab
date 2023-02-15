from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
import sparknlp
from operator import add
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, StringType, IntegerType
from pyspark.sql.functions import isnan, when, count, col, col, isnan, when, trim
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler

sc = SparkContext()
spark = (SparkSession.builder.config("spark.jars.packages","com.johnsnowlabs:spark-nlp_2.12:3.4.3").getOrCreate())

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

df = spark.read.csv("gs://bdl2022/YELP_train.csv", header=True, mode="DROPMALFORMED", schema= schema)

df.printSchema()

# Converting Nan, Null, Empty Strings to Null and then dropping the columns with any null values
def to_null(c):
    return when(~(col(c).isNull() | isnan(col(c)) | (trim(col(c)) == "")), col(c))

# Dropping Null valued rows
df = df.select([to_null(c).alias(c) for c in df.columns]).na.drop()


df = df.filter(col("stars") <= 5.0)
df = df.filter(col("stars") > 0)

# Creating the Polynomial Features
mult = udf(lambda x, y: x*y,IntegerType() )
df = df.withColumn("CoolUseful", mult("cool", "useful"))
df = df.withColumn("CoolFunny", mult("cool", "funny"))
df = df.withColumn("FunnyUseful", mult("funny", "useful"))

mult3 = udf(lambda x, y, z: x*y*z,IntegerType() )
df = df.withColumn("CoolUsefulFunny", mult3("cool", "funny", "useful"))

print("Creating the Pipeline - ")

regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
countVectors = CountVectorizer(inputCol="filtered", outputCol="cV", vocabSize=10000, minDF=5)
assembler = VectorAssembler(inputCols = ['cV', 'funny', 'useful', 'cool', "CoolUseful", \
    "CoolFunny", "FunnyUseful", "CoolUsefulFunny"], outputCol = 'features')

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, assembler])
pipelineFit = pipeline.fit(df)

dataset = pipelineFit.transform(df)
dataset.show(5)

print("Saving the Pipeline - ")
pipelineFit.save("gs://endsem/Pipeline")
print("Pipeline saved !")

print("Starting the training - ")

(trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed = 100)
lr = LogisticRegression(featuresCol="features", labelCol = 'stars')
model = lr.fit(trainingData)

print("Model Trained !")
print("-"*20)
print("Starting the evaluations - ")

predictions_train = model.transform(trainingData)
predictions_test = model.transform(testData)

eval_acc = MulticlassClassificationEvaluator(labelCol="stars", predictionCol="prediction", metricName="accuracy")
accuracy_train = eval_acc.evaluate(predictions_train)
accuracy_test = eval_acc.evaluate(predictions_test)
print("Accuracy_train: %s\nAccuarcy_test: %s\n" %(accuracy_train, accuracy_test))

eval_f1 = MulticlassClassificationEvaluator(labelCol="stars", predictionCol="prediction", metricName="f1")
f1_train = eval_f1.evaluate(predictions_train)
f1_test = eval_f1.evaluate(predictions_test)
print("F1_train: %s\nF1_test: %s\n" %(f1_train, f1_test))

print("-"*20)
print("Saving the Model - ")
model.save("gs://endsem/lr")
print("Model Saved !")