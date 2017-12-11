#!/usr/bin/env python3
"""Abstract class for model creation"""
########################################################################
# File: network.py
#  executable: network.py

# Author: Andrew Bailey
# History: 12/07/17 Created
########################################################################

from abc import ABC, abstractmethod
from basetensor.dataset import CreateTFDataset
import tensorflow as tf
import numpy as np


class CreateTFNetwork(ABC):
    def __init__(self, dataset):
        assert isinstance(dataset, CreateTFDataset)
        self.dataset = dataset
        self.summaries = []
        self.accuracy = "accuracy"
        self.loss = "loss"
        self.predict = "predict"
        super(CreateTFNetwork, self).__init__()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_ops(self):
        pass

    @staticmethod
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

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        source: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
        """
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            summary = tf.summary.scalar('mean', mean)
            self.summaries.append(summary)

if __name__ == "__main__":
    print("This is a library file. Nothing to execute")
    raise SystemExit
