from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression, OneVsRest, LinearSVC, DecisionTreeClassifier
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StandardScaler
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

sc = SparkContext()
spark = SparkSession(sc)
    
schema = schema = StructType([
    StructField("Sepal_length", DoubleType()),
    StructField("Sepal_width", DoubleType()),
    StructField("Petal_length", DoubleType()),
    StructField("Petal_width", DoubleType()),
    StructField("Class", StringType())
])

raw_data = spark.read.csv("gs://lab7_saarthak/iris.csv", header=False, mode="DROPMALFORMED", schema=schema)

indexer = StringIndexer(inputCol="Class", outputCol="label", handleInvalid='keep')

train, val = raw_data.randomSplit([0.8, 0.2])

assembler = VectorAssembler(inputCols = ["Sepal_length", "Sepal_width", "Petal_length", "Petal_width"], outputCol = "features")

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd = True, withMean = False)

# logistic Regression
lr = LogisticRegression(maxIter = 25, regParam = 0.01)

ovr = OneVsRest(classifier = lr)

p = Pipeline(stages = [indexer, assembler, scaler, ovr])

model = p.fit(train)

# pre - tuning performance
predictions_train = model.transform(train)
predictions_val = model.transform(val)

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol="prediction",metricName = "accuracy")

# compute the classification error on train and valid data.
accuracy_train = evaluator.evaluate(predictions_train)
accuracy_val = evaluator.evaluate(predictions_val)

print("-"*50)
print("Train Accuracy= " + str(accuracy_train))
print("Valid Accuracy= " + str(accuracy_val))
print("-"*50)

paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.0001,0.001,0.01,0.1,0.5]).build()

crossval = CrossValidator(estimator = p, estimatorParamMaps = paramGrid, evaluator = MulticlassClassificationEvaluator(),numFolds = 3)
cvmodel = crossval.fit(train)
prediction = cvmodel.transform(val)


evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol="prediction",metricName = "accuracy") 

#Print the Evaluation Result

model_accuracy_best= evaluator.evaluate(prediction)

print("-"*50)
print("Best_model valid performance (LR): "+ str(model_accuracy_best))
print("-"*50)

best_Model = cvmodel.bestModel

best_Model.save('gs://lab7_saarthak/trained_model')

