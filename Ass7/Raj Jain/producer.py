from kafka import KafkaProducer
from google.cloud import storage

client = storage.Client()
bucket = client.get_bucket("rajlab7")

blob = bucket.get_blob("iris.csv")
x = blob.download_as_string()

x = x.decode('utf-8')
producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda k: k.encode('utf-8'))

data = x.split("\n")

for row in data:
    producer.send("lab7",row)
    producer.flush()
