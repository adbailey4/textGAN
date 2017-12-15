#!/usr/bin/env python
"""Train models for textgan"""
########################################################################
# File: train_textgan.py
#  executable: train_textgan.py
#
# Author: Andrew Bailey
# History: 12/08/17 Created
########################################################################

import sys
import os
from datetime import datetime
import argparse
import tensorflow as tf
from basetensor.utils import optimistic_restore, test_for_nvidia_gpu
from basetensor.abstract import GanTFTraining
from textgan.tweet_datasets import TweetGeneratorDataset, TweetDiscriminatorDataset
from textgan.models import TweetGenerator, TweetDiscriminator, TextGan
from py3helpers.utils import create_logger, DotDict, list_dir, load_json, save_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CommandLine(object):
    """
    Handle the command line, usage and help requests.

    attributes:
    myCommandLine.args is a dictionary which includes each of the available
    command line arguments as myCommandLine.args['option']

    methods:
    do_usage_and_die()
    prints usage and help and terminates with an error.

    """

    def __init__(self, in_opts=None):
        """CommandLine constructor.

        Implements a parser to interpret the command line argv string using
        argparse.
        """
        # define program description, usage and epilog
        self.parser = argparse.ArgumentParser(description='TweetGAN: if you load one config file the program will train'
                                                          'on that model description. If gan config and one other '
                                                          'config file is supplied then it will load a pretrained '
                                                          'model from the other config file.',
                                              epilog="Dont forget to tar the files",
                                              usage='%(prog)s use "-h" for help')

        # optional arguments
        self.parser.add_argument('-c', '--gan_config',
                                 help='path to json gan config', required=True)
        self.parser.add_argument('-d', '--discriminator_config',
                                 help='path to json discriminator config', required=True)
        self.parser.add_argument('-g', '--generator_config',
                                 help='path to json generator config', required=True)
        self.parser.add_argument('-v', '--verbose',
                                 help='verbose option', default=False, action="store_true")
        self.parser.add_argument('--debug',
                                 help='More print statements specifying location in program',
                                 default=False, action="store_true")

        # allow optional arguments not passed by the command line
        if in_opts is None:
            self.args = vars(self.parser.parse_args())
        elif type(in_opts) is list:
            self.args = vars(self.parser.parse_args(in_opts))
        else:
            self.args = in_opts

    def do_usage_and_die(self, message):
        """ Print string and usage then return 2

        If a critical error is encountered, where it is suspected that the
        program is not being called with consistent parameters or data, this
        method will write out an error string (str), then terminate execution
        of the program.
        """
        print(message, file=sys.stderr)
        self.parser.print_help(file=sys.stderr)
        return 2


class GanTraining(GanTFTraining):
    """Give options for pretraining and training textGAN"""

    def __init__(self, gen_model, dis_model, log, gan_model=None):
        self.dis_model = dis_model
        self.gen_model = gen_model
        self.log = log
        self.gan_model = gan_model
        super(GanTraining, self).__init__([gen_model, dis_model])

    def train_gan(self, params, gen_params, dis_params):
        """Tran Gan"""
        d_learning_rate = tf.train.exponential_decay(params.learning_rate, self.gen_model.global_step,
                                                     100000, 0.96, staircase=True)

        g_learning_rate = tf.train.exponential_decay(params.learning_rate, self.dis_model.global_step,
                                                     100000, 0.96, staircase=True)

        d_opt = tf.train.AdamOptimizer(learning_rate=d_learning_rate)
        g_opt = tf.train.AdamOptimizer(learning_rate=g_learning_rate)

        # define update step
        print(self.gan_model.d_vars)
        print(self.gan_model.g_vars)
        train_d = d_opt.minimize(self.gan_model.d_loss, var_list=self.gan_model.d_vars,
                                 global_step=self.dis_model.global_step)
        train_g = g_opt.minimize(self.gan_model.g_loss, var_list=self.gan_model.g_vars,
                                 global_step=self.gen_model.global_step)
        print("Defined Optimizers", file=sys.stderr)
        config = tf.ConfigProto(log_device_placement=False,
                                intra_op_parallelism_threads=8,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(self.gen_model.dataset.pretrain_iterator.initializer)
            sess.run(self.gen_model.dataset.random_iterator.initializer)
            real_tweets_handle = sess.run(self.dis_model.dataset.real_iterator.string_handle())
            sess.run(self.dis_model.dataset.fake_iterator.initializer)
            sess.run(self.dis_model.dataset.real_iterator.initializer)

            if params.load_model:
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.restore(sess, params.model_path)
                saver.save(sess, params.model_path,
                           global_step=self.gen_model.global_step, write_meta_graph=True)
                print("Loaded Model: {}".format(params.model_path), file=sys.stderr)
            else:
                sess.run(tf.global_variables_initializer())
                if gen_params.load_model:
                    optimistic_restore(sess, gen_params.model_path)
                    print("Using weights from pre-trained Generator", file=sys.stderr)
                if dis_params.load_model:
                    optimistic_restore(sess, dis_params.model_path)
                    print("Using weights from pre-trained Discriminator", file=sys.stderr)
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)

            if not dis_params.load_model and not gen_params.load_model:
                # initialize
                sess.run(tf.global_variables_initializer())
                # save meta graph
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.save(sess, params.model_path,
                           global_step=self.gen_model.global_step, write_meta_graph=True)
                print("Saving Model: {}".format(params.model_path), file=sys.stderr)
            writer = tf.summary.FileWriter(params.trained_model_dir, sess.graph)
            # start training
            print("Started Training", file=sys.stderr)
            step = 0
            display = 0
            while step < params.iterations:
                _, dloss_real, d_acc_real = sess.run([train_d, self.gan_model.d_loss, self.gan_model.d_accuracy],
                                                     feed_dict={self.gan_model.use_generator: False,
                                                                self.dis_model.dataset.handle: real_tweets_handle})
                _, _, dloss_fake, gloss, d_acc_fake, g_acc = sess.run(
                    [train_g, train_d, self.gan_model.d_loss, self.gan_model.g_loss,
                     self.gan_model.d_accuracy, self.gan_model.g_accuracy],
                    feed_dict={self.gan_model.use_generator: True,
                               self.dis_model.dataset.handle: real_tweets_handle})
                step += 3

                while g_acc < 0.5 and step // 100 <= display:
                    _, g_acc = sess.run([train_g, self.gan_model.g_accuracy],
                                        feed_dict={self.gan_model.use_generator: True,
                                                   self.dis_model.dataset.handle: real_tweets_handle})

                    step += 1
                while (d_acc_real + d_acc_fake) / 2.0 < 0.5 and step // 100 <= display:
                    _, d_acc_real = sess.run([train_d, self.gan_model.d_accuracy],
                                             feed_dict={self.gan_model.use_generator: False,
                                                        self.dis_model.dataset.handle: real_tweets_handle})
                    _, dloss_fake, d_acc_fake, g_acc = sess.run(
                        [train_d, self.gan_model.d_loss,
                         self.gan_model.d_accuracy, self.gan_model.g_accuracy],
                        feed_dict={self.gan_model.use_generator: True,
                                   self.dis_model.dataset.handle: real_tweets_handle})

                    step += 2

                if step // 100 > display:
                    display += 1
                    g_summary, g_step, fake_tweets = sess.run(
                        [self.gan_model.generator_summary, self.gen_model.global_step,
                         self.gan_model.passed_sentences],
                        feed_dict={self.gan_model.use_generator: True,
                                   self.dis_model.dataset.handle: real_tweets_handle})
                    d_summary, d_step = sess.run(
                        [self.gan_model.discriminator_summary_real, self.dis_model.global_step],
                        feed_dict={self.gan_model.use_generator: False,
                                   self.dis_model.dataset.handle: real_tweets_handle})
                    writer.add_summary(d_summary, global_step=d_step)

                    d_summary, d_step = sess.run(
                        [self.gan_model.discriminator_summary_fake, self.dis_model.global_step],
                        feed_dict={self.gan_model.use_generator: True,
                                   self.dis_model.dataset.handle: real_tweets_handle})
                    writer.add_summary(g_summary, global_step=g_step)
                    writer.add_summary(d_summary, global_step=d_step)
                    print("Global Discriminator Step: {}".format(d_step))
                    print("Global Generator Step: {}".format(g_step))
                    if len(fake_tweets) != 0:
                        sentence = ''.join([self.gen_model.dataset.ix_to_char[x] for x in fake_tweets[0]])
                        try:
                            print(
                                "step {}: ".format(g_step) + repr(sentence[:sentence.index(params.end_tweet_char) + 1]),
                                file=open(params.model_path + ".tweets", "a"))
                        except ValueError:
                            print(repr(sentence))
                            print("step {}: ".format(g_step) + repr(sentence),
                                  file=open(params.model_path + ".tweets", "a"))
                    saver.save(sess, params.model_path,
                               global_step=self.gen_model.global_step + self.dis_model.global_step,
                               write_meta_graph=False)

            print("Finished Training", file=sys.stderr)

    def run_generator(self, params, gen_params):
        "Run trained generator"
        config = tf.ConfigProto(log_device_placement=False,
                                intra_op_parallelism_threads=8,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(self.gen_model.dataset.pretrain_iterator.initializer)
            sess.run(self.gen_model.dataset.random_iterator.initializer)
            # real_tweets_handle = sess.run(self.dis_model.dataset.real_iterator.string_handle())
            # sess.run(self.dis_model.dataset.fake_iterator.initializer)
            # sess.run(self.dis_model.dataset.real_iterator.initializer)

            if params.load_model:
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.restore(sess, params.model_path)
                saver.save(sess, params.model_path,
                           global_step=self.gen_model.global_step, write_meta_graph=True)
                print("Loaded Model: {}".format(params.model_path), file=sys.stderr)
            elif gen_params.load_model:
                sess.run(tf.global_variables_initializer())
                optimistic_restore(sess, gen_params.model_path)
                print("Using weights from pre-trained Generator", file=sys.stderr)
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
            else:
                # initialize
                sess.run(tf.global_variables_initializer())
                # save meta graph
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.save(sess, params.model_path,
                           global_step=self.gen_model.global_step, write_meta_graph=True)
                print("Saving Model: {}".format(params.model_path), file=sys.stderr)
            writer = tf.summary.FileWriter(params.trained_model_dir, sess.graph)
            # start training
            print("Start Generating Fake Tweets", file=sys.stderr)
            step = 0
            display = 0
            while step < params.iterations:
                prediction, g_step = sess.run([self.gen_model.predict, self.gen_model.global_step])
                for tweet in prediction:
                    sentence = ''.join([self.gen_model.dataset.ix_to_char[x] for x in tweet])
                    try:
                        print(repr(sentence))
                        print(
                            "step {}: ".format(g_step) + repr(sentence[:sentence.index(params.end_tweet_char) + 1]),
                            file=open(gen_params.model_path + ".inference.tweets", "a"))
                    except ValueError:
                        print(repr(sentence))
                        print("step {}: ".format(g_step) + repr(sentence),
                              file=open(gen_params.model_path + ".inference.tweets", "a"))

            print("Finished Training", file=sys.stderr)

    def read_parameters(self):
        pass

    def pretrain_generator(self, params):
        """Pretrain text generator"""
        learning_rate = tf.train.exponential_decay(params.learning_rate, self.gen_model.global_step,
                                                   100000, 0.96, staircase=True)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = opt.minimize(self.gen_model.loss, var_list=self.gen_model.trainable_vars,
                                global_step=self.gen_model.global_step)
        all_summary = tf.summary.merge_all()
        print("Defined Optimizer", file=sys.stderr)
        config = tf.ConfigProto(log_device_placement=True,
                                intra_op_parallelism_threads=8,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(self.gen_model.dataset.pretrain_iterator.initializer)

            if params.load_model:
                writer = tf.summary.FileWriter(params.trained_model_dir, sess.graph)
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.restore(sess, params.model_path)
                saver.save(sess, params.model_path,
                           global_step=self.gen_model.global_step, write_meta_graph=True)
                print("Loaded Model: {}".format(params.model_path), file=sys.stderr)
            else:
                # initialize
                writer = tf.summary.FileWriter(params.trained_model_dir, sess.graph)
                sess.run(tf.global_variables_initializer())
                # save meta graph
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.save(sess, params.model_path,
                           global_step=self.gen_model.global_step, write_meta_graph=True)
                print("Saving Model: {}".format(params.model_path), file=sys.stderr)

            # start training
            print("Started Training", file=sys.stderr)
            step = 0
            while step < params.iterations:
                _, gloss = sess.run([train_op, self.gen_model.loss])
                step += 1
                if step % 100 == 0:
                    summary_info, g_step, fake_tweets = sess.run([all_summary, self.gen_model.global_step,
                                                                  self.gen_model.predict])
                    print("Step {}: Generator Loss {}".format(g_step, gloss))
                    sentence = ''.join([self.gen_model.dataset.ix_to_char[x] for x in fake_tweets[0]])
                    try:
                        print("step {}: ".format(g_step) + repr(sentence[:sentence.index(params.end_tweet_char) + 1]),
                              file=open(params.model_path + ".tweets", "a"))
                    except ValueError:
                        print(repr(sentence))
                        print("step {}: ".format(g_step) + repr(sentence),
                              file=open(params.model_path + ".tweets", "a"))

                    writer.add_summary(summary_info, global_step=g_step)
                    saver.save(sess, params.model_path,
                               global_step=self.gen_model.global_step, write_meta_graph=False)

        print("Finished Training", file=sys.stderr)

    def pretrain_discriminator(self, params):
        """Pretrain discriminator network"""
        learning_rate = tf.train.exponential_decay(params.learning_rate, self.dis_model.global_step,
                                                   100000, 0.96, staircase=True)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = opt.minimize(self.dis_model.loss, var_list=self.dis_model.trainable_vars,
                                global_step=self.dis_model.global_step)
        all_summary = tf.summary.merge_all()
        print("Defined Optimizer", file=sys.stderr)
        config = tf.ConfigProto(log_device_placement=False,
                                intra_op_parallelism_threads=8,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            fake_tweets_handle = sess.run(self.dis_model.dataset.fake_iterator.string_handle())
            real_tweets_handle = sess.run(self.dis_model.dataset.real_iterator.string_handle())
            sess.run(self.dis_model.dataset.real_iterator.initializer)
            sess.run(self.dis_model.dataset.fake_iterator.initializer)

            if params.load_model:
                writer = tf.summary.FileWriter(params.trained_model_dir, sess.graph)
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.restore(sess, params.model_path)
                saver.save(sess, params.model_path,
                           global_step=self.dis_model.global_step, write_meta_graph=True)
                print("Loaded Model: {}".format(params.model_path), file=sys.stderr)
            else:
                # initialize
                writer = tf.summary.FileWriter(params.trained_model_dir, sess.graph)
                sess.run(tf.global_variables_initializer())
                # save meta graph
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.save(sess, params.model_path,
                           global_step=self.dis_model.global_step, write_meta_graph=True)
                print("Saving Model: {}".format(params.model_path), file=sys.stderr)

            # start training
            print("Started Training", file=sys.stderr)
            step = 0
            while step < params.iterations:
                # z_batch = np.random.normal(0, 1, size=[batch_size, len_x])
                _, d_loss = sess.run([train_op, self.dis_model.loss],
                                     feed_dict={self.dis_model.dataset.handle: real_tweets_handle})
                _, d_loss = sess.run([train_op, self.dis_model.loss],
                                     feed_dict={self.dis_model.dataset.handle: fake_tweets_handle})

                step += 1
                if step % 100 == 0:
                    summary_info, d_step = sess.run([all_summary, self.dis_model.global_step],
                                                    feed_dict={self.dis_model.dataset.handle: fake_tweets_handle})
                    print("Step {}: Discriminator Loss {}".format(d_step, d_loss))
                    writer.add_summary(summary_info, global_step=d_step)
                    saver.save(sess, params.model_path,
                               global_step=self.dis_model.global_step, write_meta_graph=False)

        print("Finished Training", file=sys.stderr)


def load_gan_params(config_path, name, create_dir=True):
    """Load parameters from config json file"""
    assert os.path.isfile(config_path), "Config file does not exist"
    params = DotDict(load_json(config_path)[name])
    # check arguments and define

    if params.load_model:
        if params.use_checkpoint:
            model_path = tf.train.latest_checkpoint(params.trained_model_dir)
            params.model_path = model_path
        else:
            model_path = params.model_path
        assert os.path.isfile(model_path + ".index"), "No index file associated with model path"
        save_json({name: params}, params.model_path + ".config.json")
    else:
        params.trained_model_dir = os.path.join(params.save_model_dir, datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
        params.model_path = os.path.join(params.trained_model_dir, params.model_name)
        if create_dir:
            try:
                os.makedirs(params.trained_model_dir)
            except OSError:
                pass
            params.load_model = True
            params.use_checkpoint = True
            save_json({name: params}, params.model_path + ".config.json")
            params.load_model = False
            params.use_checkpoint = False

    return params


def main():
    ##################################
    command_line = CommandLine()
    gen_config_path = command_line.args["generator_config"]
    gan_config_path = command_line.args["gan_config"]
    dis_config_path = command_line.args["discriminator_config"]

    assert dis_config_path, "Must specify Discriminator config"
    # check arguments and define
    params = load_gan_params(gan_config_path, name="gan")
    dis_params = load_gan_params(dis_config_path, name="discriminator", create_dir=False)
    gen_params = load_gan_params(gen_config_path, name="generator", create_dir=False)
    print("Generator Model Path: {}".format(gen_params.model_path), file=sys.stderr)
    print("Discriminator Model Path: {}".format(dis_params.model_path), file=sys.stderr)
    print("GAN Model Path: {}".format(params.model_path), file=sys.stderr)
    save_json(dict(generator=gen_params), params.model_path + "generator.config.json")
    save_json(dict(discriminator=dis_params), params.model_path + "discriminator.config.json")

    # gan_config_path = command_line.args["gan_config"]
    log1 = create_logger(params.model_path, name="a", info=command_line.args["verbose"],
                         debug=command_line.args["debug"])

    file_list = list_dir(params.twitter_data_path, ext="csv")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # gpus = test_for_nvidia_gpu(2)
    # print(",".join(gpus))
    tweets = TweetGeneratorDataset(log=log1)
    len_x, max_seq_len, ix_to_char, char_to_ix, all_tweets, all_seq_len = \
        tweets.load_training_data(file_list, reduction_level=params.reduction_level, end_tweet_char=params.end_tweet_char)
    tweets.create_dataset(batch_size=params.batch_size, n_epochs=1, pretrain=False)
    tweets.create_iterator()
    tweets.test()


    d_tweets = TweetDiscriminatorDataset(log=log1)
    d_tweets.end_tweet_char = params.end_tweet_char
    d_tweets.len_x = len_x
    d_tweets.max_seq_len = max_seq_len
    d_tweets.ix_to_char = ix_to_char
    d_tweets.char_to_ix = char_to_ix
    d_tweets.all_tweets = all_tweets
    d_tweets.all_seq_len = all_seq_len
    d_tweets.create_dataset(batch_size=params.batch_size, n_epochs=1, pretrain=False)
    d_tweets.create_iterator()
    d_tweets.test()
    with tf.device('/gpu:0'):
        gen_model = TweetGenerator(tweets, gen_params.layers, log=log1)
        if False:
            gen_model.create_model(pretrain=False)
            gen_model.create_ops()
            gan_training = GanTraining(gen_model, gen_model, log=log1)
            gan_training.run_generator(params, gen_params)

    with tf.device('/gpu:1'):
        d_model = TweetDiscriminator(d_tweets, log=log1, layers=dis_params.layers)

    gan_model = TextGan(generator=gen_model, discriminator=d_model, log=log1)
    gan_model.create_model()
    gan_model.create_ops()
    gan_training = GanTraining(gen_model=gen_model, dis_model=d_model, log=log1, gan_model=gan_model)
    gan_training.train_gan(params, dis_params=dis_params, gen_params=gen_params)


if __name__ == '__main__':
    main()
