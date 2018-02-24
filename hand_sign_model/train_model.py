import tensorflow as tf

from hand_sign_model.data_loader import data_generator
from hand_sign_model.utils import logger, config
from hand_sign_model.trainers import trainer
from hand_sign_model.models import model

configs = config.process_config('./configs/example.json')

with tf.Session() as sess:
    model = model.AlexNet(config=configs)
    data = data_generator.DataGenerator(config=configs)
    logger = logger.Logger(sess=sess, config=configs)
    trainer = trainer.AlexNetTrainer(sess=sess, model=model, data=data, config=configs, logger=logger)
    trainer.train()
