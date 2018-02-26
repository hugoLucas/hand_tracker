import tensorflow as tf


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.data = None

    def next_batch(self, batch_size):
        batch = self.data.batch(batch_size)
        iterator = batch.make_one_shot_iterator()
        return iterator.get_next()

    def configure_dataset(self, source_record, repeat=None):
        self.data = tf.data.TFRecordDataset(filenames=source_record)
        self.data = self.data.map(self.load_image_from_record)
        self.data.repeat(repeat)
        self.data = self.data.shuffle(buffer_size=self.config.num_elements)

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


class TrainingDataGenerator(DataGenerator):
    def __init__(self, config):
        super(TrainingDataGenerator, self).__init__(config)
        self.configure_dataset(source_record=self.config.train_tfrecord)


class TestDataGenerator(DataGenerator):
    def __init__(self, config):
        super(TestDataGenerator, self).__init__(config)
        self.configure_dataset(source_record=self.config.test_tfrecord)
