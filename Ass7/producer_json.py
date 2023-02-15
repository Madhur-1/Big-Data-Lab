import json
from json import loads
from csv import DictReader

from kafka import KafkaProducer


def producer():
    producer = KafkaProducer(bootstrap_servers='10.188.0.2:9092',api_version=(0,11,5))
    topicName = 'iris-data'

    i = 0
    with open('iris_data.csv') as f:
        lister = DictReader(f)
        for row in lister:
            if i != 0:
                for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
                    row[col] = float(row[col])
                print(row)
                ack = producer.send(topicName, json.dumps(row).encode('utf-8'))
                metadata = ack.get()
                print(metadata.topic, metadata.partition)
            i += 1

if __name__ == '__main__':
    producer()