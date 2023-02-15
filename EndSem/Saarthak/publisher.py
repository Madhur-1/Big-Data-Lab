from kafka import KafkaProducer
from google.cloud import storage
#from time import sleep

client = storage.Client()
bucket = client.get_bucket("yelp-test")
#fetch test dataset blobs
test_blobs = []
for blob in bucket.list_blobs():
	if 'test_yelp_dataset/part' in blob.name:
		test_blobs.append(blob)

for blob in test_blobs:
	x = blob.download_as_string()
	x = x.decode('utf-8')
	producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda k: k.encode('utf-8'))
	data = x.split("\n")
	for row in data:
	    producer.send("KafkaYelp",row)
	    print(row)
	    producer.flush()

