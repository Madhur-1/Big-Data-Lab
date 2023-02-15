import base64
from google.cloud import storage

def hello_pubsub(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    print('Message:', pubsub_message)
    client = storage.Client()
    bucket = client.get_bucket('krazy-bucket')
    blob = bucket.get_blob(pubsub_message)
    x = blob.download_as_string()
    x = x.decode('utf-8')
    x = x.split('\n')
    print('Number of lines:', len(x))
