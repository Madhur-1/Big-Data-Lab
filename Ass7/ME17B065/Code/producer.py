import csv
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_server='10.128.0.11:9902', value_serializer=lambda v: json.dumps(v).encode('utf-8'))
topicName = 'lab7-test'

i = 0
with open('Iris.csv', 'rt') as f:
    reader = csv.reader(f, dialect=csv.excel)
    for row in reader:
        if i != 0: # exclude header
            res = producer.send(topicName, row)
            print(res,get(timeout=60))
        i += 1
