from pyspark import SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as f
from itertools import chain
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("ML_predictor").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

ip_address = "10.142.0.2:2181" 
df = spark.readStream.format("kafka").option("kafka.bootstrap.servers",ip_address).option("subscribe","lab7").load()

#set header
split_cols = f.split(df.value,',')
df = df.withColumn('Sepal_length',split_cols.getItem(0))
df = df.withColumn('Sepal_width',split_cols.getItem(1))
df = df.withColumn('Petal_length',split_cols.getItem(2))
df = df.withColumn('Petal_width',split_cols.getItem(3))
df = df.withColumn('Class',split_cols.getItem(4))

for col in ['Sepal_length','Sepal_width','Petal_length','Petal_width']:
    df = df.withColumn(col,df[col].cast('float'))

df.createOrReplaceTempView('iris')

model = PipelineModel.load('gs://lab7_saarthak/trained_model')

#assembler = VectorAssembler(inputCols = ["Sepal_length", "Sepal_width", "Petal_length", "Petal_width"], outputCol = "features")

#df = assembler.transform(df)
df = df.withColumn('true_label', df['Class'])

mapping = dict(zip(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],[0.0,1.0,2.0]))
mapping_expr = f.create_map([f.lit(x) for x in chain(*mapping.items())])
df = df.withColumn('Class', mapping_expr[f.col("Class")])

predictions = model.transform(df)

mapping = dict(zip([0.0,1.0,2.0], ['Iris-setosa','Iris-versicolor','Iris-virginica']))
mapping_expr = f.create_map([f.lit(x) for x in chain(*mapping.items())])
output_df = predictions.withColumn('prediction', mapping_expr[f.col("prediction")])[['prediction','true_label']]

output_df = output_df.withColumn('correct', f.when((f.col('prediction')=='Iris-setosa') & (f.col('true_label')=='Iris-setosa'),1).when((f.col('prediction')=='Iris-versicolor') & (f.col('true_label')=='Iris-versicolor'),1).when((f.col('prediction')=='Iris-virginica') & (f.col('true_label')=='Iris-virginica'),1).otherwise(0))

df_acc = output_df.select(f.format_number(f.avg('correct')*100,2).alias('accuracy'))

output_df2 = output_df[['prediction','true_label','correct']]
output_df2.createOrReplaceTempView('output')
query = output_df2.writeStream.queryName("output").outputMode('update').format('console').start()
query2 = df_acc.writeStream.outputMode('complete').format('console').start()
#spark.sql("select avg(correct) from output").show()
query.awaitTermination()
query2.awaitTermination()