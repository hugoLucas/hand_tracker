import tensorflow as tf

from hand_sign_model.data_loader import data_generator
from hand_sign_model.utils import logger, config
from hand_sign_model.trainers import trainer
from hand_sign_model.models import model

configs = config.process_config('./configs/example.json')
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

model = model.AlexNet(config=configs)
data = data_generator.DataGenerator(config=configs)
input_x, input_y = data.next_batch(batch_size=10)

with tf.Session() as sess:
    sess.run(init_op)

    val1, val2 = sess.run([input_x, input_y])

    logger = logger.Logger(sess=sess, config=configs)
    feed_dict = {model.x: val1, model.y: val2, model.is_training: True}
    _, loss, acc = sess.run([model.train_step, model.cross_entropy, model.accuracy], feed_dict=feed_dict)
