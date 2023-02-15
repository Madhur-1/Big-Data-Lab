from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression, LinearSVC, OneVsRest, DecisionTreeClassifier, RandomForestClassifier, MultilayerPerceptronClassifier, NaiveBayes
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.classification import SVMWithSGD
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext()
spark = SparkSession(sc)

iris_data = spark.read.format("bigquery").option("table", "iris_dataset.iris_data_table").load()
iris_data.createOrReplaceTempView("iris")


def vector_from_inputs(r):
    target = 0
    if r["Species"] == "Iris-versicolor":
        target = 0
    elif r["Species"] == "Iris-virginica":
        target = 1
    else:
        target = 2
    return (float(target), Vectors.dense(float(r["SepalLengthCm"]),
                                  float(r["SepalWidthCm"]),
                                  float(r["PetalLengthCm"]),
                                  float(r["PetalWidthCm"])))


iris_data = iris_data.rdd.map(vector_from_inputs).toDF(["label",
                                                        "features"])

training_data, test_data = iris_data.randomSplit([0.8, 0.2])

# lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)
# lsvc = LinearSVC(maxIter=10, regParam=0.1, tol=1E-6, fitIntercept=True)
# ovr = OneVsRest(classifier=lsvc)
# dt = DecisionTreeClassifier()
# rf = RandomForestClassifier(numTrees=5, seed=42)
# nb = NaiveBayes(modelType="gaussian")
layers = [4, 5, 3]
nn = MultilayerPerceptronClassifier(maxIter=100, layers=layers, seed=1234)
model = nn.fit(training_data)

predictions = model.transform(training_data)
test_predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
test_accuracy = evaluator.evaluate(test_predictions)

print("Train Accuracy = %g" % (accuracy))
print("Test Accuracy = %g" % (test_accuracy))





