import tensorflow as tf

from hand_sign_model.data_loader import data_generator
from hand_sign_model.utils import config
from hand_sign_model.models import model


graph_to_load = "/home/hugolucas/PycharmProjects/ml_hands/experiments/example/checkpoint/my_model-2020"
checkpoint_dir = "/home/hugolucas/PycharmProjects/ml_hands/experiments/example/checkpoint"

configs = config.process_config('./configs/example.json')
data = data_generator.TestDataGenerator(config=configs)
model = model.AlexNet(config=configs)

saver = tf.train.Saver()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.initialize_all_variables())
with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, graph_to_load)

    running_avg, n = 0, 1
    while True:
        _x, _y = data.next_batch(batch_size=configs.batch_size)
        input_x, input_y = sess.run([_x, _y])

        feed_dict = {model.x: input_x, model.y: input_y, model.is_training: False}
        acc = sess.run(model.accuracy, feed_dict=feed_dict)
        running_avg += acc
        n += 1

        print(running_avg/n, acc)
