import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_path = '/home/hugolucas/PycharmProjects/ml_hands/hand_sign_gen/train.tfrecord'  # address to save the hdf5 file

with tf.Session() as sess:
    feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}

    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)

    image = tf.decode_raw(features['image'], tf.float32)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image, [300, 300, 3])

    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
                                            min_after_dequeue=10)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(1):
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        for j in range(6):
            plt.subplot(2, 3, j + 1)
            plt.imshow(img[j, ...])
            plt.title('cat' if lbl[j] == 0 else 'dog')
        plt.show()

    coord.request_stop()
    coord.join(threads)
    sess.close()
