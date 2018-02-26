from hand_sign_model.base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class AlexNetTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, test_data, config, logger):
        super(AlexNetTrainer, self).__init__(sess, model, train_data, config, logger)
        self.test_data = test_data
        self.epoch_number = self.model.global_step_tensor.eval(self.sess)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        loss_history, accuracy_history = [], []

        train_batch_x, train_batch_y = self.data.next_batch(self.config.batch_size)
        test_batch_x, test_batch_y = self.test_data.next_batch(self.config.batch_size)

        for iter in loop:
            loss, acc = self.train_step(train_batch_x, train_batch_y)
            loss_history.append(loss)
            accuracy_history.append(acc)

            loop.set_description('Train iteration {}.{}'.format(self.epoch_number, iter))

        summaries_dict = {
            'loss': np.mean(loss_history),
            'acc': np.mean(accuracy_history),
        }

        self.epoch_number = self.model.global_step_tensor.eval(self.sess)
        if self.epoch_number % 100 == 0:
            test_set_accuracy = self.test_set_eval(test_batch_x, test_batch_y)
            summaries_dict['test_acc'] = np.mean(test_set_accuracy)

        self.logger.summarize(self.epoch_number, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self, input_x, input_y):
        input_x, input_y = self.sess.run([input_x, input_y])
        feed_dict = {self.model.x: input_x, self.model.y: input_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def test_set_eval(self, input_batch_x, input_batch_y):
        progress = tqdm(range(0, self.config.test_iters))

        accuracy_history = []
        for iter in progress:
            input_x, input_y = self.sess.run([input_batch_x, input_batch_y])
            feed_dict = {self.model.x: input_x, self.model.y: input_y, self.model.is_training: False}
            acc = self.sess.run(self.model.accuracy, feed_dict=feed_dict)
            accuracy_history.append(acc)

            progress.set_description('Test iteration {}'.format(iter))

        return accuracy_history