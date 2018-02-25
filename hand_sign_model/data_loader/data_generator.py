import tensorflow as tf


class DataGenerator():
    def __init__(self, config):
        self.config = config
        self.dataset = tf.data.TFRecordDataset(filenames=config.train_tfrecord)
        self.dataset = self.dataset.map(self.load_image_from_record)
        self.dataset.repeat(1)
        self.dataset = self.dataset.shuffle(buffer_size=config.num_elements)

    def next_batch(self, batch_size):
        batch = self.dataset.batch(batch_size)
        iterator = batch.make_one_shot_iterator()
        return iterator.get_next()

    def load_image_from_record(self, serialized):
        features = {
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        }
        parsed_example = tf.parse_single_example(serialized, features)

        image = tf.decode_raw(parsed_example['image'], tf.float32)
        image = tf.reshape(image, [300, 300, 3])

        label = parsed_example['label']
        label = tf.one_hot(indices=label, depth=self.config.num_logits)

        return image, label
