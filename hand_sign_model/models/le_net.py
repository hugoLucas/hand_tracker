from hand_sign_model.base.base_model import BaseModel
import tensorflow as tf


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        # self.y = tf.placeholder(tf.float32, shape=[None, 10])
        #
        # # network_architecture
        # d1 = tf.layers.dense(self.x, 512, activation=tf.nn.relu, name="densee2")
        # d2 = tf.layers.dense(d1, 10)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.input_size)

        with tf.name_scope("convolutional_layers"):
            self.conv_1 = tf.layers.conv2d(
                inputs=self.x,
                filters=96,
                kernel_size=[11, 11],
                padding="SAME",
                strides=4,
                activation=tf.nn.relu
            )
            self.pool_2 = tf.layers.max_pooling2d(
                inputs=self.conv_1,
                pool_size=[3, 3],
                strides=2,
                padding='VALID'
            )
            self.conv_3 = tf.layers.conv2d(
                inputs=self.pool_2,
                filters=256,
                kernel_size=[5, 5],
                padding="SAME",
                strides=1,
                activation=tf.nn.relu
            )
            self.pool_4 = tf.layers.max_pooling2d(
                inputs=self.conv_3,
                pool_size=[3, 3],
                strides=2,
                padding='VALID'
            )
            self.conv_5 = tf.layers.conv2d(
                inputs=self.pool_4,
                filters=384,
                kernel_size=[3, 3],
                padding="SAME",
                strides=1,
                activation=tf.nn.relu
            )
            self.conv_6 = tf.layers.conv2d(
                inputs=self.conv_5,
                filters=384,
                kernel_size=[3, 3],
                padding="SAME",
                strides=1,
                activation=tf.nn.relu
            )
            self.conv_7 = tf.layers.conv2d(
                inputs=self.conv_6,
                filters=256,
                kernel_size=[3, 3],
                padding="SAME",
                strides=1,
                activation=tf.nn.relu
            )

        with tf.name_scope("connected_layers"):
            flatten_8 = tf.reshape(self.conv_7, [self.config.batch_size, -1])
            layer_8 = tf.layers.dense(inputs=flatten_8, units=4096, activation=tf.nn.relu)
            self.dense_8 = tf.layers.dropout(inputs=layer_8, rate=0.5, training=self.is_training)
            layer_9 = tf.layers.dense(inputs=self.dense_8, units=4096, activation=tf.nn.relu)
            self.dense_9 = tf.layers.dropout()

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

