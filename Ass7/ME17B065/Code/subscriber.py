from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.ml.classification import MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row



spark = SparkSession.builder.appName("Species Predicter").getOrCreate()

spark.sparkContext.setLogLevel('WARN')

model_path = 'gs://krazy-bucket/models/nn-model/'

model = MultilayerPerceptronClassificationModel.load(model_path)

print('Model Loaded....')

df = spark.readStream.format('kafka').option('kafka.bootstrap.servers', '10.128.0.11:9092').option('subscribe', 'lab7-test').load()

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
    pred = predict(vec)
    return pred


def predict(x):
    result = model.transform(x).head().prediction
    return 1
 
query = df.writeStream.foreach(formVector).outputMode('update').format('console').start()

#query = vectors \
#    .writeStream \
#    .outputMode("complete") \
#    .format("console") \
#    .start()
query.awaitTermination()

print('Data loaded...')

