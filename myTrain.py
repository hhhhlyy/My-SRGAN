import tensorflow as tf
from os import makedirs
from numpy import array, load
from scipy.misc import imresize
from os.path import exists

from myModel import Model
from utils import BatchGenerator
import plot
class Trainer(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
        print("dictionary here:")
        print(dictionary)
        print('Importing training set ...')
        run_flags = dictionary['flags']
        self.dataset = load(file=run_flags.data, allow_pickle=False)
        self.datasize = self.dataset.shape[0]
        print('Done.')

    def train(self):
        run_flags = self.flags
        sess = self.sess
        sw = self.sw

        hr_img = tf.placeholder(tf.float32, [None, run_flags.input_width * run_flags.scale,
                                             run_flags.input_length * run_flags.scale, 3]) #128*128*3 as default
        lr_img = tf.placeholder(tf.float32, [None, run_flags.input_width,
                                             run_flags.input_length, 3])#64*64*3 as default
        myModel = Model(locals())

        out_gen = Model.generative(myModel, lr_img)

        real_out_dis = Model.discriminative(myModel, hr_img)

        fake_out_dis = Model.discriminative(myModel, out_gen, reuse=True)

        cost_gen, cost_dis, var_train, var_gen, var_dis = \
        Model.costs_and_vars(myModel, hr_img, out_gen, real_out_dis, fake_out_dis)

        # cost_gen, cost_dis, var_train, var_gen, var_dis = \
        # Model.wgan_loss(myModel, hr_img, out_gen, real_out_dis, fake_out_dis)

        optimizer_gen = tf.train.AdamOptimizer(learning_rate=run_flags.lr_gen). \
            minimize(cost_gen, var_list=var_gen)
        optimizer_dis = tf.train.AdamOptimizer(learning_rate=run_flags.lr_dis). \
            minimize(cost_dis, var_list=var_dis)

        init = tf.global_variables_initializer()

        with sess:
            sess.run(init)

            saver = tf.train.Saver()

            if not exists('models'):
                makedirs('models')

            passed_iters = 0

            for epoch in range(1, run_flags.epochs + 1):
                print('Epoch:', str(epoch))

                for batch in BatchGenerator(run_flags.batch_size, self.datasize):
                    batch_hr = self.dataset[batch] / 255.0
                    batch_lr = array([imresize(img, size=(run_flags.input_width, run_flags.input_length, 3)) \
                                      for img in batch_hr])

                    _, gc, dc = sess.run([optimizer_gen, cost_gen, cost_dis],
                                         feed_dict={hr_img : batch_hr, lr_img: batch_lr})
                    sess.run([optimizer_dis],
                             feed_dict={hr_img : batch_hr, lr_img: batch_lr})

                    passed_iters += 1

                    if passed_iters % run_flags.sample_iter == 0:
                        print('Passed iterations=%d, Generative cost=%.9f, Discriminative cost=%.9f' % \
                              (passed_iters, gc, dc))
                        plot.plot('train_dis_cost_gan', abs(dc))
                        plot.plot('train_gen_cost_gan', abs(gc))

                    if (passed_iters < 5) or (passed_iters % 100 == 99):
                        plot.flush()

                    plot.tick()

                if run_flags.checkpoint_iter and epoch % run_flags.checkpoint_iter == 0:
                    saver.save(sess, '/'.join(['models', run_flags.model, run_flags.model]))

                    print('Model \'%s\' saved in: \'%s/\'' \
                          % (run_flags.model, '/'.join(['models', run_flags.model])))

            print('Optimization finished.')

            saver.save(sess, '/'.join(['models', run_flags.model, run_flags.model]))

            print('Model \'%s\' saved in: \'%s/\'' \
                  % (run_flags.model, '/'.join(['models', run_flags.model])))