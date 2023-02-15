from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.functions import isnan, when, count, col, col, isnan, when, trim
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler

sc = SparkContext()
spark = (SparkSession.builder.config("spark.jars.packages","com.johnsnowlabs:spark-nlp_2.12:3.4.3").getOrCreate())

df = spark.read.csv("gs://bdl2022/YELP_train.csv", header=True, mode="DROPMALFORMED")

df.printSchema()

# Converting Nan, Null, Empty Strings to Null and then dropping the columns with any null values
def to_null(c):
    return when(~(col(c).isNull() | isnan(col(c)) | (trim(col(c)) == "")), col(c))


df = df.select([to_null(c).alias(c) for c in df.columns]).na.drop()

df = df.withColumn("cool", col("cool").cast("int"))
df = df.withColumn("funny", col("funny").cast("int"))
df = df.withColumn("stars", col("stars").cast("int"))
df = df.withColumn("useful", col("useful").cast("int"))

df = df.filter(col("stars") <= 5.0)
df = df.filter(col("stars") > 0)
df = df.filter(df["cool"] <= 10.0)
df = df.filter(df["cool"] >= 0.0)
df = df.filter(col("useful") <= 10.0)
df = df.filter(col("useful") >= 0.0)
df = df.filter(col("funny") <= 10.0)
df = df.filter(col("funny") >= 0.0)


regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
countVectors = CountVectorizer(inputCol="filtered", outputCol="cV", vocabSize=10000, minDF=5)
CleanData = VectorAssembler(inputCols = ['cV'], outputCol = 'features')

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, CleanData])
pipelineFit = pipeline.fit(df)

dataset = pipelineFit.transform(df)
dataset.show(5)

(trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed = 100)
nb = NaiveBayes(featuresCol = 'features', labelCol = 'stars', smoothing=1.0, modelType="multinomial")
model = nb.fit(trainingData)

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

