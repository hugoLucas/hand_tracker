import tensorflow as tf


class DataGenerator():
    def __init__(self, config):
        self.config = config
        self.dataset = tf.data.TFRecordDataset(filenames=config.train_tfrecord)
        self.dataset = self.dataset.map(self.load_image_from_record)
        self.dataset.repeat(None)
        self.dataset = self.dataset.shuffle(buffer_size=config.num_elements)

    def next_batch(self, batch_size):
        batch = self.dataset.batch(batch_size)
        iterator = batch.make_one_shot_iterator()
        images, labels = iterator.get_next()
        yield images, labels


    @staticmethod
    def load_image_from_record(serialized):
        features = {
            'label': tf.FixedLenFeature([], tf.string),
            'img': tf.FixedLenFeature([], tf.string)
        }
        parsed_example = tf.parse_single_example(serialized, features)

        image = parsed_example['img']
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)

        label = parsed_example['label']

        return image, label

