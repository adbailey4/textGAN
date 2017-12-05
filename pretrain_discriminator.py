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


from run_gan import pretrain_discriminator, load_tweet_data, list_dir, TrainingData, variable_summaries

# reduction level definitions
RL_NONE = 0
RL_LOW = 1
RL_MED = 2
RL_HIGH = 3


def main():
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
    model_name = "pretrain_discriminator"
    output_dir = os.path.abspath("models/pretrain_discriminator")
    twitter_data_path = os.path.abspath("example_tweet_data/train_csv")
    trained_model_dir = os.path.abspath("models/pretrain_discriminator")

    load_model = True

    if load_model:
        model_path = tf.train.latest_checkpoint(trained_model_dir)
        # model_path = "models/test_gan/first_pass_gan-9766-19678"
    else:
        model_path = os.path.join(output_dir, model_name)
    log.info("Model Path {}".format(model_path))

    ##################################

    file_list = list_dir(twitter_data_path, ext="csv")

    len_x, max_seq_len, ix_to_char, char_to_ix, all_tweets, all_seq_len = load_tweet_data(file_list,
                                                                                          reduction_level=reduction_level)

    # print(len(words), prob, n_swaps)
    len_y = 1
    # create placeholders for discriminator
    place_X = tf.placeholder(tf.float32, shape=[None, max_seq_len, len_x], name='Input')
    place_Y = tf.placeholder(tf.float32, shape=[None, len_y], name='Label')
    place_Seq = tf.placeholder(tf.int32, shape=[None], name='Sequence_Length')
    # define discriminator and generator global steps
    d_global_step = tf.get_variable(
        'd_global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # create easily accessible training data
    training_data = TrainingData(all_tweets, all_seq_len, len_x, batch_size, char_to_ix=char_to_ix,
                                 end_tweet_char=end_tweet_char, gen_pretrain=False, dis_pretrain=True)
    training_data.start_threads(n_threads=threads)
    # training_data.stop_threads()
    # x,s,y = training_data.get_batch()
    # print(y.shape)
    # discriminator for twitter data
    Dx = pretrain_discriminator(place_X, place_Seq, d_n_hidden, len_y, forget_bias=forget_bias, reuse=False)
    log.info("Discriminator Model Built")

    # discriminator accuracy
    d_accuracy = tf.reduce_mean(tf.cast(tf.equal(Dx, place_Y), tf.float32))
    # calculate loss
    d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=place_Y))
    log.info("Created Loss function")

    # create summary info
    variable_summaries(d_loss, "discriminator_loss")
    variable_summaries(d_accuracy, "discriminator_accuracy")
    all_summary = tf.summary.merge_all()

    # partition trainable variables
    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if 'pretrain_d_lstm' in var.name]

    # define optimizers
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # define update step
    trainerD = opt.minimize(d_loss, var_list=d_vars, global_step=d_global_step)
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
            write_graph = True
            log.info("Loaded Model: {}".format(trained_model_dir))
        else:
            # initialize
            writer = tf.summary.FileWriter(output_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
            saver.save(sess, model_path,
                       global_step=d_global_step)
            write_graph = True

        # start training
        log.info("Started Training")
        step = 0
        while step < iterations:
            x_batch, seq_batch, y_batch = training_data.get_batch()
            # z_batch = np.random.normal(0, 1, size=[batch_size, len_x])
            _, dLoss, dAccuracy= sess.run([trainerD, d_loss, d_accuracy], feed_dict={place_X: x_batch,
                                                               place_Seq: seq_batch,
                                                               place_Y: y_batch})
            # print(dPredict)
            # print(dAccuracy)
            # print(dLoss)
            # z_batch = np.random.normal(0, 1, size=[batch_size, len_x])
            step += 1
            if step % 100 == 0:
                summary_info, d_step = sess.run([all_summary, d_global_step], feed_dict={place_X: x_batch,
                                                                                         place_Seq: seq_batch,
                                                                                         place_Y: y_batch})
                print("Step {}: Discriminator Loss {}".format(d_step, dLoss))
                writer.add_summary(summary_info, global_step=d_step)
                saver.save(sess, model_path,
                           global_step=d_global_step, write_meta_graph=write_graph)
                write_graph = False

    training_data.stop_threads()
    log.info("Finished Training")



if __name__ == '__main__':
    main()
