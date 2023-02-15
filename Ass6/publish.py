def publish(data, context):
    from google.cloud import pubsub_v1
    publisher = pubsub_v1.PublisherClient()
    topic_path = 'projects/big-data-lab-madhur/topics/lab6'
    file_name = data['name']
    print('------- Triggered by addition of a file -------')
    print('The name of the file is: ', file_name)
        # Publishes a message
    try:
        publish_future = publisher.publish(topic_path, data=bytes(file_name, 'utf-8'))
        publish_future.result()  # Verify the publish succeeded
        return 'Message published.'
    except Exception as e:
        print(e)
        return (e, 500)