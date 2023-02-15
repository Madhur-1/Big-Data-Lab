from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler

sc = SparkContext()
spark = SparkSession(sc)

data = spark.read.format("bigquery").option("table", "iris_data.iris_data").load()
# data.createOrReplaceTempView("iris")

def vector_from_inputs(r):
    label = 0
    if r['class'] == 'Iris-virginica':
        label = 1
    elif r['class'] == 'Iris-versicolor':
        label = 2
    return (label, Vectors.dense(float(r['sepal_length']), 
                                    float(r['sepal_width']),
                                    float(r['petal_length']), 
                                    float(r['petal_width'])))


df = data.rdd.map(vector_from_inputs).toDF(['label', 'features'])

# No Pre-processing

print("------------------------------")
print("NO PREPROCESSING - RAW DATA \n")

df_no_pre = df.select('*')
train, test = df_no_pre.randomSplit([0.8, 0.2])
train.cache()

# Logistic Regression
lr = LogisticRegression(fitIntercept=True, maxIter=10, regParam=0.3)
lrModel = lr.fit(train)
lrModel.save('gs://big-data-lab-madhurj/Ass7/Model')

lr_preds_tr = lrModel.transform(train)
lr_preds_tst = lrModel.transform(test)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
lr_acc_tr = evaluator.evaluate(lr_preds_tr)
lr_acc_tst = evaluator.evaluate(lr_preds_tst)

print("Logistic Regression :")
print("Train Acc: ", lr_acc_tr)
print("Test Acc: ", lr_acc_tst)
print("\n")

# Decision Tree Classifier

dt = DecisionTreeClassifier()
dtModel = dt.fit(train)

dt_preds_tr = dtModel.transform(train)
dt_preds_tst = dtModel.transform(test)
dt_acc_tr = evaluator.evaluate(dt_preds_tr)
dt_acc_tst = evaluator.evaluate(dt_preds_tst)

print("Decision Tree Classifier :")
print("Train Acc: ", dt_acc_tr)
print("Test Acc: ", dt_acc_tst)
print("\n")

# Random Forest CLassifier

rf = RandomForestClassifier(numTrees=5, seed=42)
rfModel = rf.fit(train)
rfModel.write().overwrite().save('gs://big-data-lab-madhurj/Ass7/RandomForestModel')

rf_preds_tr = rfModel.transform(train)
rf_preds_tst = rfModel.transform(test)
rf_acc_tr = evaluator.evaluate(rf_preds_tr)
rf_acc_tst = evaluator.evaluate(rf_preds_tst)

print("Random Forest Classifier :")
print("Train Acc: ", rf_acc_tr)
print("Test Acc: ", rf_acc_tst)
print("\n")

# Naive Bayes Classifier

nb = NaiveBayes(modelType='gaussian')
nbModel = nb.fit(train)

nb_preds_tr = nbModel.transform(train)
nb_preds_tst = nbModel.transform(test)
nb_acc_tr = evaluator.evaluate(nb_preds_tr)
nb_acc_tst = evaluator.evaluate(nb_preds_tst)

print("Naive Bayes Classifier :")
print("Train Acc: ", nb_acc_tr)
print("Test Acc: ", nb_acc_tst)
print("\n")

## Using Standard Scaler

print("------------------------------")
print("STANDARD SCALER \n")

df_standard = df.select('*')
scaler = StandardScaler(inputCol="features", outputCol='Scaledfeatures')
scalerModel = scaler.fit(df_standard)
scaledData = scalerModel.transform(df_standard)
df_scld = scaledData.select('label', 'Scaledfeatures')
df_scaled = df_scld.withColumnRenamed("Scaledfeatures", "features")

train, test = df_scaled.randomSplit([0.8, 0.2])
train.cache()

# Logistic Regression
lr = LogisticRegression(fitIntercept=True, maxIter=10, regParam=0.3)
lrModel = lr.fit(train)

lr_preds_tr = lrModel.transform(train)
lr_preds_tst = lrModel.transform(test)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
lr_acc_tr = evaluator.evaluate(lr_preds_tr)
lr_acc_tst = evaluator.evaluate(lr_preds_tst)

print("Logistic Regression :")
print("Train Acc: ", lr_acc_tr)
print("Test Acc: ", lr_acc_tst)
print("\n")

# Decision Tree Classifier

dt = DecisionTreeClassifier()
dtModel = dt.fit(train)

dt_preds_tr = dtModel.transform(train)
dt_preds_tst = dtModel.transform(test)
dt_acc_tr = evaluator.evaluate(dt_preds_tr)
dt_acc_tst = evaluator.evaluate(dt_preds_tst)

print("Decision Tree Classifier :")
print("Train Acc: ", dt_acc_tr)
print("Test Acc: ", dt_acc_tst)
print("\n")

# Random Forest CLassifier

rf = RandomForestClassifier(numTrees=5, seed=42)
rfModel = rf.fit(train)

rf_preds_tr = rfModel.transform(train)
rf_preds_tst = rfModel.transform(test)
rf_acc_tr = evaluator.evaluate(rf_preds_tr)
rf_acc_tst = evaluator.evaluate(rf_preds_tst)

print("Random Forest Classifier :")
print("Train Acc: ", rf_acc_tr)
print("Test Acc: ", rf_acc_tst)
print("\n")

# Naive Bayes Classifier

nb = NaiveBayes(modelType='gaussian')
nbModel = nb.fit(train)

nb_preds_tr = nbModel.transform(train)
nb_preds_tst = nbModel.transform(test)
nb_acc_tr = evaluator.evaluate(nb_preds_tr)
nb_acc_tst = evaluator.evaluate(nb_preds_tst)

print("Naive Bayes Classifier :")
print("Train Acc: ", nb_acc_tr)
print("Test Acc: ", nb_acc_tst)
print("\n")
