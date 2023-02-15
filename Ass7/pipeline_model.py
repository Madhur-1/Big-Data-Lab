from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler,  StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StringType, FloatType

sc = SparkContext()
spark = SparkSession(sc)

schema = StructType()\
    .add("sepal_length", FloatType())\
    .add("sepal_width", FloatType())\
    .add("petal_length", FloatType())\
    .add("petal_width", FloatType())\
    .add("class", StringType())

# data = spark.read.format("bigquery").option("table", "iris_data.iris_data").schema(schema).load()
data = spark.read.csv("gs://big-data-lab-madhurj/Ass7/iris_data.csv", header=True, mode="DROPMALFORMED", schema=schema)

data.printSchema()
data.show()
# data.createOrReplaceTempView("iris")
stage_1 = StringIndexer(inputCol= 'class', outputCol= 'label', handleInvalid='keep')
vec_ass = VectorAssembler(inputCols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                          outputCol='features')
lr = LogisticRegression(fitIntercept=True, maxIter=10, regParam=0.3)

pipeline = Pipeline(stages=[stage_1, vec_ass, lr])

# fit the pipeline model and transform the data as defined
pipeline_model = pipeline.fit(data)
pipeline_model.save('gs://big-data-lab-madhurj/Ass7/Pipeline_Model')