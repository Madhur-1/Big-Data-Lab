import csv, json
from kafka import KafkaProducer


def producer():
    producer = KafkaProducer(bootstrap_servers='10.188.0.2:9092',api_version=(0,11,5), value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    topicName = 'lab7'

    i = 0
    with open('iris_data.csv') as f:
        lister = csv.reader(f, dialect=csv.excel)
        for row in lister:
            if i != 0: 
                print(row)
                res = producer.send(topicName, row)
                print(res)
            i += 1

if __name__ == '__main__':
    producer()