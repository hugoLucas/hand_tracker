from hand_sign_model.base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class AlexNetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(AlexNetTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        loss_history, accurary_history = [], []

        for it in loop:
            loss, acc = self.train_step()
            loss_history.append(loss)
            accurary_history.append(acc)
        loss = np.mean(loss_history)
        acc = np.mean(accurary_history)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {'loss': loss, 'acc': acc}
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        input_x = self.sess.run(batch_x)
        feed_dict = {self.model.x: input_x, self.model.y: input_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
