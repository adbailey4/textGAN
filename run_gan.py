#!/usr/bin/env python
"""Create text Gan"""
########################################################################
# File: run_gan.py
#  executable: run_gan.py
#
# Author: Andrew Bailey
# History: 11/28/17 Created
########################################################################
from __future__ import print_function
import logging as log
import sys
import os
import re
import subprocess
import collections
from timeit import default_timer as timer
import json
import argparse
from datetime import datetime
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.client import timeline
import unicodecsv
from unidecode import unidecode
import threading

try:
    import Queue as queue
except ImportError:
    import queue

# reduction level definitions
RL_NONE = 0
RL_LOW = 1
RL_MED = 2
RL_HIGH = 3

# character mapping for emojis
BASIC_EMOJIS = [u'\U0001F600', u'\U0001F61B', u'\U0001F620', u'\U0001F62D',  # grin, angry, tongue, cry
                u'\U0001F618', u'\u2764', u'\U0001F609', u'\U0001F60D',  # blow kiss, heart, wink, heart eyes
                u'\u2639', u'\U0001F4A9', u'\U0001F44D', u'\U0001F60E',  # frown, poop, thumbs up, sunglasses
                u'\U0001F610', u'\U0001F44C', u'\u2611', u'\U0001F525', ]  # neutral face, ok, check mark, fire
BASIC_EMOJI_ENUMERATIONS = [[u'<emoji_{}'.format(i), BASIC_EMOJIS[i]] for i in range(len(BASIC_EMOJIS))]

# do you want to live life on the edge?  then leave this line uncommented, you badass!
import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

training_labels = collections.namedtuple('training_data', ['input', 'seq_len'])


def list_dir(path, ext=""):
    """get all file paths from local directory with extension"""
    if ext == "":
        only_files = [os.path.join(os.path.abspath(path), f) for f in \
                      os.listdir(path) if \
                      os.path.isfile(os.path.join(os.path.abspath(path), f))]
    else:
        only_files = [os.path.join(os.path.abspath(path), f) for f in \
                      os.listdir(path) if \
                      os.path.isfile(os.path.join(os.path.abspath(path), f)) \
                      if f.split(".")[-1] == ext]
    return only_files


# this prints the following error:
#   "RuntimeWarning: Surrogate character u'\udc23' will be ignored. You might be using a narrow Python build"
def reduce_unicode_characters(unicode_str, reduction_level=RL_MED):
    if reduction_level == RL_HIGH:
        return unidecode(unicode_str)
    if reduction_level == RL_MED:
        for substitution in BASIC_EMOJI_ENUMERATIONS:
            unicode_str = unicode_str.replace(substitution[1], substitution[0])
        unicode_str = unidecode(unicode_str)
        for substitution in BASIC_EMOJI_ENUMERATIONS:
            unicode_str = unicode_str.replace(substitution[0], substitution[1])
        return unicode_str
    return unicode_str


def read_tweet_data(filename, reduction_level):
    """
    Read in tweet collection of "HillaryClinton" and "realDonaldTrump" (or "none in case of test data).
    Other twitter handles would break this function.
    :param filename: File to read tweet data from.
    :return: list with handles and list with tweets; in order of file appearance.
    """
    fileH = open(filename, 'r')
    header = fileH.readline().rstrip().split(',')
    handles = []
    tweets = []
    goodHandles = ["HillaryClinton", "realDonaldTrump", "none"]
    r = unicodecsv.reader(fileH, encoding='utf-8')
    for row in r:
        handles.append(row[0])
        tweets.append(reduce_unicode_characters(row[1], reduction_level=reduction_level))
    # write to file to test if data is read in correctly (should be exactly the same as the input file)
    # outfileH = open('./out.csv','w')
    # outfileH.write(",".join(header) + "\n")
    # for i in range(0, len(handles)):
    #   outfileH.write(handles[i] + "," + tweets[i]+"\n")
    return handles, tweets


def load_tweet_data(file_list, reduction_level, end_tweet_char=u'\u26D4'):
    """Read in tweet data from csv file
    source: https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/
    """
    all_tweets = []
    all_seq_len = []
    all_vectorized_tweets = []
    # collect all tweets from csv files
    for tweet_file in file_list:
        _, tweets = read_tweet_data(tweet_file, reduction_level=reduction_level)
        all_tweets.extend(tweets)
        all_seq_len.extend([len(tweet) + 1 for tweet in tweets])
    # changed to sorted so it is deterministic
    chars = (list(set(''.join(all_tweets))))
    # get all the possible characters and the maximum tweet length
    # plus 1 for ending character
    assert end_tweet_char not in chars, "Sentence Ending was used. Change to new unicode character"
    chars.append(end_tweet_char)
    chars = sorted(chars)
    print("Characters in Corpus\n", repr(''.join(chars)))
    print("Number of Characters: {}".format(len(chars)))
    len_x = len(chars)
    seq_len = max(all_seq_len)
    print("Max seq length: {}".format(seq_len))
    # create translation dictionaries
    ix_to_char = {ix: char for ix, char in enumerate(chars)}
    char_to_ix = {char: ix for ix, char in enumerate(chars)}

    return len_x, seq_len, ix_to_char, char_to_ix, all_tweets, all_seq_len


def generator(input_vector, max_seq_len, n_hidden, batch_size, dropout=False, output_keep_prob=1):
    """Feeds output from lstm into input of same lstm cell"""
    with tf.variable_scope("generator_lstm"):
        # make new lstm cell for every step
        def make_cell(input_vector, state, dropout):
            cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=forget_bias)
            if dropout and output_keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=output_keep_prob)
            return cell(inputs=input_vector, state=state)

        state = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=forget_bias).zero_state(batch_size, tf.float32)
        outputs = []
        for time_step in range(max_seq_len):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = make_cell(input_vector, state, dropout=dropout)
            outputs.append(cell_output)
            input_vector = cell_output
        # print(outputs)
        output = tf.stack(outputs, 1)

    return output


def fulconn_layer(input_data, output_dim, seq_len=1, activation_func=None):
    """Create a fully connected layer.
    source:
    https://stackoverflow.com/questions/39808336/tensorflow-bidirectional-dynamic-rnn-none-values-error/40305673
    """
    # get input dimensions
    input_dim = int(input_data.get_shape()[1])
    weight = tf.get_variable(name="weights", shape=[input_dim, output_dim * seq_len],
                             initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / (2 * output_dim))))
    bias = tf.get_variable(name="bias", shape=[output_dim * seq_len],
                           initializer=tf.zeros_initializer)

    # weight = tf.Variable(tf.random_normal([input_dim, output_dim * seq_len]), name="weights")
    # bias = tf.Variable(tf.random_normal([output_dim * seq_len]), name="bias")
    if activation_func:
        output = activation_func(tf.nn.bias_add(tf.matmul(input_data, weight), bias))
    else:
        output = tf.nn.bias_add(tf.matmul(input_data, weight), bias)
    return output, weight, bias


def discriminator(input_vector, sequence_length_placeholder, n_hidden, len_y, forget_bias=5, reuse=False):
    """Feeds output from lstm into input of same lstm cell"""
    with tf.variable_scope("discriminator_lstm", reuse=reuse):
        rnn_layers = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        output, _ = tf.nn.dynamic_rnn(cell=rnn_layers,
                                      inputs=input_vector,
                                      dtype=tf.float32,
                                      time_major=False,
                                      sequence_length=sequence_length_placeholder)

        batch_size = tf.shape(output)[0]
        last_outputs = tf.gather_nd(output, tf.stack([tf.range(batch_size), sequence_length_placeholder - 1], axis=1))

        with tf.variable_scope("final_full_conn_layer", reuse=tf.AUTO_REUSE):
            final_output, weights, bias = fulconn_layer(input_data=last_outputs, output_dim=len_y)

        return final_output


summaries = []


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    source: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
    """
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        summary = tf.summary.scalar('mean', mean)
        summaries.append(summary)


class TrainingData(object):
    """Get batches from training dataset"""

    def __init__(self, all_tweets, all_seq_len, len_x, batch_size):
        self.all_tweets = all_tweets
        self.all_seq_len = all_seq_len
        self.len_x = len_x
        self.max_seq_len = max(all_seq_len)
        self.len_data = len(all_tweets)
        self.batch_size = batch_size
        self.stop_event = threading.Event()
        self.queue = queue.Queue(maxsize=20)

    def get_batch(self):
        """Get batch of data"""
        batch = self.queue.get()
        return batch[0], batch[1]

    def read_in_batchs(self, stop_event):
        """Read in data as needed by the batch"""
        while not stop_event.is_set():
            x_batch = []
            seq_batch = []
            for i in range(self.batch_size):
                vector_tweet = np.zeros([self.max_seq_len, self.len_x])
                for indx, char in enumerate(np.random.choice(self.all_tweets)):
                    vector_tweet[indx, char_to_ix[char]] = 1
                # add tweet ending character to tweet
                vector_tweet[indx + 1, char_to_ix[end_tweet_char]] = 1
                x_batch.append(vector_tweet)
                seq_batch.append(indx + 2)

            self.queue.put([np.asarray(x_batch), np.asarray(seq_batch)])

    def start_threads(self, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.read_in_batchs, args=(self.stop_event,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

    def stop_threads(self):
        """Kill daemon threads if needed"""
        self.stop_event.set()


##################################
# define hyperparameters
log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
end_tweet_char = u'\u26D4'
batch_size = 10
d_n_hidden = 100
forget_bias = 1
learning_rate = 0.001
iterations = 1000
threads = 4
reduction_level = RL_HIGH
model_name = "ascii_test"
# output_dir = "/Users/andrewbailey/CLionProjects/nanopore-RNN/textGAN/models/test_gan"
output_dir = os.path.abspath("models/ascii_test")
# twitter_data_path = "/Users/andrewbailey/CLionProjects/nanopore-RNN/textGAN/example_tweet_data/train_csv"
# trained_model_dir = "/Users/andrewbailey/CLionProjects/nanopore-RNN/textGAN/models/test_gan"
twitter_data_path = os.path.abspath("example_tweet_data/train_csv")
trained_model_dir = os.path.abspath("models/ascii_test")

load_model = False
if load_model:
    model_path = tf.train.latest_checkpoint(trained_model_dir)
    # model_path = "models/test_gan/first_pass_gan-9766-19678"
else:
    model_path = os.path.join(output_dir, model_name)
log.info("Model Path {}".format(model_path))

##################################


file_list = list_dir(twitter_data_path, ext="csv")

len_x, max_seq_len, ix_to_char, char_to_ix, all_tweets, all_seq_len = load_tweet_data(file_list, reduction_level=reduction_level)

stop_char_index = tf.get_variable('stop_char_index', [],
                                  initializer=tf.constant_initializer(char_to_ix[end_tweet_char]),
                                  trainable=False, dtype=tf.int64)

max_seq_len_tensor = tf.get_variable('max_seq_len', [],
                                     initializer=tf.constant_initializer(max_seq_len),
                                     trainable=False, dtype=tf.int32)

# right now we are not passing generator output through fully conn layer so we have to match the size of each character
gen_n_hidden = len_x
len_y = 1
# create placeholders for discriminator
place_X = tf.placeholder(tf.float32, shape=[None, max_seq_len, len_x], name='Input')
place_Seq = tf.placeholder(tf.int32, shape=[None], name='Sequence_Length')
# random input for generator
place_Z = tf.placeholder(tf.float32, shape=[None, len_x], name='Label')
# define discriminator and generator global steps
g_global_step = tf.get_variable(
    'g_global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)
d_global_step = tf.get_variable(
    'd_global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)

# create easily accessible training data
training_data = TrainingData(all_tweets, all_seq_len, len_x, batch_size)
training_data.start_threads(n_threads=threads)
# x_batch, seq_batch = training_data.get_batch()
# training_data.stop_threads()

# create models
Gz = generator(place_Z, max_seq_len, gen_n_hidden, batch_size)
log.info("Generator Model Built")


def index1d(t):
    """Get index of first appearance of specific character"""
    index = tf.cast(tf.reduce_min(tf.where(tf.equal(stop_char_index, t))), dtype=tf.int32)
    # return index
    return tf.cond(index < 0, lambda: tf.cast(max_seq_len_tensor, dtype=tf.int32),
                   lambda: tf.cast(tf.add(index, 1), dtype=tf.int32))


# get character indexes for all sequences
gen_char_index = tf.argmax(Gz, axis=2)
# length of the sequence for the generator network based on termination character
z_seq_length = tf.map_fn(index1d, gen_char_index, dtype=tf.int32, back_prop=False)
# discriminator for generator output
Dg = discriminator(Gz, z_seq_length, d_n_hidden, len_y, forget_bias=forget_bias, reuse=False)
log.info("1st Discriminator Model Built")

# discriminator for twitter data
Dx = discriminator(place_X, place_Seq, d_n_hidden, len_y, forget_bias=forget_bias, reuse=True)
log.info("2nd Discriminator Model Built")

# generator sentences
g_predict = tf.reshape(tf.argmax(Gz, 2), [batch_size, 1, max_seq_len])

# generator accuracy
g_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.sign(Dg), tf.ones_like(Dg)), tf.float32))

# discriminator accuracy
d_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.sign(tf.stack([Dg, Dx])),
                                             tf.stack([tf.negative(tf.ones_like(Dg)), tf.ones_like(Dx)])), tf.float32))

# sentences that passed the discriminator
indices = tf.where(tf.equal(tf.sign(Dg), tf.ones_like(Dg)))
passed_sentences = tf.gather_nd(g_predict, indices)

# calculate loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake
log.info("Created Loss functions")

# create summary info
variable_summaries(g_loss, "generator_loss")
variable_summaries(d_loss, "discriminator_loss")
variable_summaries(g_accuracy, "generator_accuracy")
variable_summaries(d_accuracy, "discriminator_accuracy")
all_summary = tf.summary.merge_all()

# partition trainable variables
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'discriminator_lstm' in var.name]
g_vars = [var for var in tvars if 'generator_lstm' in var.name]

# define optimizers
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

# define update step
trainerD = opt.minimize(d_loss, var_list=d_vars, global_step=d_global_step)
trainerG = opt.minimize(g_loss, var_list=g_vars, global_step=g_global_step)
log.info("Defined Optimizers")

# define config
config = tf.ConfigProto(log_device_placement=False,
                        intra_op_parallelism_threads=8,
                        allow_soft_placement=True)

with tf.Session(config=config) as sess:
    if load_model:
        writer = tf.summary.FileWriter(trained_model_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
        saver.restore(sess, model_path)
        write_graph = False
        log.info("Loaded Model: {}".format(trained_model_dir))
    else:
        # initialize
        writer = tf.summary.FileWriter(output_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
        saver.save(sess, model_path,
                   global_step=d_global_step + g_global_step)
        write_graph = True

    # start training
    log.info("Started Training")
    step = 0
    while step < iterations:
        x_batch, seq_batch = training_data.get_batch()
        z_batch = np.random.normal(0, 1, size=[batch_size, len_x])
        if step == 0:
            _, gLoss, gAccuracy = sess.run([trainerG, g_loss, g_accuracy], feed_dict={place_Z: z_batch})
            _, dLoss = sess.run([trainerD, d_loss],
                                feed_dict={place_Z: z_batch, place_X: x_batch, place_Seq: seq_batch})
            x_batch, seq_batch = training_data.get_batch()
            z_batch = np.random.normal(0, 1, size=[batch_size, len_x])
            step += 2

        if gAccuracy < 0.2:
            _, gAccuracy = sess.run([trainerG, g_accuracy], feed_dict={place_Z: z_batch})
            step += 1
        if gAccuracy > 0.9:
            _, gAccuracy = sess.run([trainerD, g_accuracy],
                                    feed_dict={place_Z: z_batch, place_X: x_batch, place_Seq: seq_batch})
            step += 1
        else:
            _, gAccuracy = sess.run([trainerD, g_accuracy],
                                    feed_dict={place_Z: z_batch, place_X: x_batch, place_Seq: seq_batch})
            _, gAccuracy = sess.run([trainerG, g_accuracy], feed_dict={place_Z: z_batch})
            step += 2

        if step % 10 == 0:
            summary_info, d_step, g_step = sess.run([all_summary, d_global_step, g_global_step],
                                                    feed_dict={place_Z: z_batch, place_X: x_batch,
                                                               place_Seq: seq_batch})
            writer.add_summary(summary_info, global_step=d_step + g_step)
            saver.save(sess, model_path,
                       global_step=d_global_step + g_global_step, write_meta_graph=write_graph)
            write_graph = False
        if step % 100 == 0:
            fake_tweets, d_step, g_step = sess.run([passed_sentences, d_global_step, g_global_step],
                                                   feed_dict={place_Z: z_batch, place_X: x_batch, place_Seq: seq_batch})
            print("Global Discriminator Step: {}".format(d_step))
            print("Global Generator Step: {}".format(g_step))
            if len(fake_tweets) != 0:
                sentence = ''.join([ix_to_char[x] for x in fake_tweets[0]])
                try:
                    print(repr(sentence[:sentence.index(end_tweet_char) + 1]))
                except ValueError:
                    print(repr(sentence))

training_data.stop_threads()
log.info("Finished Training")
