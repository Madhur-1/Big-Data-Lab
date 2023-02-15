from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparknlp.annotator import Lemmatizer, Stemmer, Tokenizer, Normalizer
from sparknlp.base import DocumentAssembler, Finisher
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
# The imports, above, allow us to access SparkML features specific to linear
# regression as well as the Vectors types.

sc = SparkContext()
spark = (SparkSession.builder.config("spark.jars.packages","com.johnsnowlabs:spark-nlp_2.12:3.0.3").getOrCreate())


# Read the data from BigQuery as a Spark Dataframe.
yelp_dataset = spark.read.json("gs://bdl2021_final_project/yelp_train.json")

yelp_dataset.printSchema()

# Create a view so that Spark SQL queries can be run against the data.
yelp_dataset.createOrReplaceTempView("yelp")

# Set the altitude column as the target
from pyspark.ml import Pipeline


#training_data, test_data = iris_data.randomSplit([0.8,0.2], seed =42)
training_data, test_data = yelp_dataset.randomSplit([0.8,0.2], seed =42)

#Save the test dataset
test_data.write.json("gs://yelp-test/test_yelp_dataset")

# Pipeline
document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalizer")
stemmer = Stemmer().setInputCols(["normalizer"]).setOutputCol("stem")
finisher = Finisher().setInputCols(["stem"]).setOutputCols(["to_spark"]).setValueSplitSymbol(" ")
stopword_remover = StopWordsRemover(inputCol="to_spark", outputCol="filtered")
tf = CountVectorizer(inputCol="filtered", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="features")

lr = LogisticRegression(featuresCol = 'features', labelCol = 'stars', maxIter = 10)

pipe = Pipeline(
	stages = [document_assembler,tokenizer, normalizer, stemmer, finisher, stopword_remover, tf, idf, lr])

print("Fitting the model to the data")
lr_model = pipe.fit(training_data)

# Saving the model
try:
	lr_model.save("gs://yelp-test/LR_TFIDF_model_complete")
except:
	model = lr_model.stages[2]
	print(model)
	model.save("gs://yelp-test/LR_TFIDF_model_complete")

print("Saving Done")

# Evaluator
lr_evaluator = MulticlassClassificationEvaluator(predictionCol = "prediction", 
                                  labelCol="stars",
                                  metricName="f1")
#training predictions
prediction_train = lr_model.transform(training_data)
prediction_train.select("prediction", "stars", "features").show(5)
print("Evaluating on Training data:", lr_evaluator.evaluate(prediction_train))

# test prediction
prediction_test = lr_model.transform(test_data)

prediction_test.select("prediction", "stars", "features").show(5)

# evaluating the test score

print("Evaluating on Test data:", lr_evaluator.evaluate(prediction_test))