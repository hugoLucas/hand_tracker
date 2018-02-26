import tensorflow as tf

from hand_sign_model.data_loader import data_generator
from hand_sign_model.utils import logger, config
from hand_sign_model.trainers import trainer
from hand_sign_model.models import model

confs = config.process_config('./configs/example.json')

alex_model = model.AlexNet(config=confs)
train_data = data_generator.TrainingDataGenerator(config=confs)
test_data = data_generator.TestDataGenerator(config=confs)

train_x, train_y = train_data.next_batch(batch_size=confs.batch_size)
test_x, test_y = test_data.next_batch(batch_size=confs.batch_size)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.initialize_all_variables())
with tf.Session() as sess:
    logger = logger.Logger(sess=sess, config=confs)
    trainer = trainer.AlexNetTrainer(sess=sess, model=alex_model, train_data=train_data, test_data=test_data,
                                     config=confs, logger=logger)
    sess.run(init_op)
    trainer.train()
