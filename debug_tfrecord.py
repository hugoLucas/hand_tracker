import tensorflow as tf
from google.protobuf.json_format import MessageToJson

file = "/home/hugolucas/ml_data/egohands_data/models/research/object_detection/hand_detection/test.tfrecords"
# file = "/home/hugolucas/ml_data/out/test.tfrecords"
fileNum = 1
for example in tf.python_io.tf_record_iterator(file):
    jsonMessage = MessageToJson(tf.train.Example.FromString(example))
    print(jsonMessage)
    fileNum += 1

    if fileNum > 10:
        break
