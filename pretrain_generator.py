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

from run_gan import pretrain_generator, load_tweet_data, list_dir, TrainingData, variable_summaries, Hyperparameters
# reduction level definitions
RL_NONE = 0
RL_LOW = 1
RL_MED = 2
RL_HIGH = 3


def main():
    ##################################
    # define hyperparameters
    batch_size = 100
    learning_rate = 0.001
    iterations = 1000
    threads = 4
    ####
    params = Hyperparameters()
    ####
    load_model = False
    #####
    if load_model:
        model_path = tf.train.latest_checkpoint(params.g_trained_model_dir)
    else:
        model_path = os.path.join(params.g_output_dir, params.g_model_name)
    log.info("Model Path {}".format(model_path))

    ##################################

    file_list = list_dir(params.twitter_data_path, ext="csv")
    len_x, max_seq_len, ix_to_char, char_to_ix, all_tweets, all_seq_len = \
        load_tweet_data(file_list, reduction_level=params.reduction_level)

    gen_n_hidden = len_x

    # right now we are not passing generator output through fully conn layer so we have to match the size of each character
    # create placeholders for discriminator
    place_X = tf.placeholder(tf.float32, shape=[None, max_seq_len, len_x], name='Input')
    place_Y = tf.placeholder(tf.float32, shape=[None, max_seq_len, len_x], name='Label')
    place_Seq = tf.placeholder(tf.int32, shape=[None], name='Sequence_Length')
    # define generator global step
    g_global_step = tf.get_variable(
        'g_global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # create easily accessible training data
    training_data = TrainingData(all_tweets, all_seq_len, len_x, batch_size, char_to_ix=char_to_ix,
                                 end_tweet_char=params.end_tweet_char, gen_pretrain=True)
    training_data.start_threads(n_threads=threads)
    # create models
    Gz = pretrain_generator(place_X, place_Seq, params.g_layers, max_seq_len=max_seq_len, len_x=len_x,
                            forget_bias=params.g_forget_bias,
                            dropout=params.g_dropout, output_keep_prob=params.g_drop_prob)

    log.info("Generator Model Built")

    # generator sentences
    g_predict = tf.reshape(tf.argmax(Gz, 2), [batch_size, max_seq_len], name="g_predict")
    g_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Gz, 2), tf.argmax(place_Y, 2)), dtype=tf.int32),
                                name="g_accuracy")

    # calculate loss
    g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Gz, labels=place_Y))
    log.info("Created Loss functions")

    # create summary info
    variable_summaries(g_loss, "generator_loss")
    variable_summaries(g_accuracy, "generator_accuracy")
    all_summary = tf.summary.merge_all()

    # partition trainable variables
    tvars = tf.trainable_variables()
    g_vars = [var for var in tvars if 'pretrain_g_lstm' in var.name]

    # define optimizers
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # define update step
    trainerG = opt.minimize(g_loss, var_list=g_vars, global_step=g_global_step)
    log.info("Defined Optimizers")

    # define config
    config = tf.ConfigProto(log_device_placement=False,
                            intra_op_parallelism_threads=8,
                            allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        if load_model:
            writer = tf.summary.FileWriter(params.g_trained_model_dir, sess.graph)
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
            saver.restore(sess, model_path)
            write_graph = True
            log.info("Loaded Model: {}".format(trained_model_dir))
        else:
            # initialize
            writer = tf.summary.FileWriter(params.g_output_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
            saver.save(sess, model_path,
                       global_step=g_global_step)
            write_graph = True

        # start training
        log.info("Started Training")
        step = 0
        while step < iterations:
            x_batch, seq_batch, y_batch = training_data.get_batch()
            # z_batch = np.random.normal(0, 1, size=[batch_size, len_x])
            _, gLoss = sess.run([trainerG, g_loss], feed_dict={place_X: x_batch,
                                                               place_Seq: seq_batch,
                                                               place_Y: y_batch})
            # print(gPredict)
            # print(gAccuracy)
            # print(gLoss)
            # z_batch = np.random.normal(0, 1, size=[batch_size, len_x])
            step += 1
            if step % 100 == 0:
                summary_info, g_step = sess.run([all_summary, g_global_step], feed_dict={place_X: x_batch,
                                                                                         place_Seq: seq_batch,
                                                                                         place_Y: y_batch})
                print("Step {}: Generator Loss {}".format(g_step, gLoss))
                writer.add_summary(summary_info, global_step=g_step)
                saver.save(sess, model_path,
                           global_step=g_global_step, write_meta_graph=write_graph)
                write_graph = False

    training_data.stop_threads()
    log.info("Finished Training")


if __name__ == '__main__':
    main()
