def count_lines(data, context):
    import os
    from google.cloud import pubsub_v1
    print('Starting cloud function...')
    publisher = pubsub_v1.PublisherClient()
    topic_name = 'projects/{project_id}/topics/{topic}'.format(
    project_id='inlaid-computer-305711',topic='lab6')
    file_name = data['name']
    print('Topic name:', topic_name)
    print('File name:', file_name)
    future = publisher.publish(topic_name, bytes(file_name, 'utf-8'))
    future.result() 
