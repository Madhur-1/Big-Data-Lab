import base64
from google.cloud import storage

def subscribe(event, context):
    message = base64.b64decode(event['data']).decode('utf-8')
    print('------- Triggered by a PubSub Message -------')
    print('The message from PubSub is: ', message)
    client = storage.Client()
    bucket = client.get_bucket('bucket-lab6')
    blob = bucket.get_blob(message)
    x = blob.download_as_string()
    print("Number of lines: ", len(x.decode('utf-8').split('\n')))
