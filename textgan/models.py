#!/usr/bin/env python
"""Create text Gan"""
########################################################################
# File: models.py
#  executable: models.py
#
# Author: Andrew Bailey
# History: 12/08/17 Created
########################################################################

import sys
import tensorflow as tf
from basetensor.abstract import CreateTFNetwork

train_op = "train ops"
inference_op = "inference ops"
accuracy = "accuracy"
loss = "loss"


class TweetGenerator(CreateTFNetwork):
    """Create tweet generator using LSTMs"""

    def __init__(self, dataset, layers, log):
        super(TweetGenerator, self).__init__(dataset)
        self.layers = layers
        self.lstmcells = "lstmcells"
        self.global_step = tf.get_variable(
            'g_global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        self.final_output = "final output"
        self.trainable_vars = "vars"
        self.pretrain = True
        self.log = log
        self.stop_char_index = tf.get_variable('stop_char_index', [],
                                               initializer=tf.constant_initializer(
                                                   self.dataset.char_to_ix[self.dataset.end_tweet_char]),
                                               trainable=False, dtype=tf.int64)
        self.max_seq_len_tensor = tf.get_variable('max_seq_len', [],
                                                  initializer=tf.constant_initializer(self.dataset.max_seq_len),
                                                  trainable=False, dtype=tf.int32)

        self.p_x, self.p_seq, self.p_y = self.dataset.pretrain_iterator.get_next()
        self.x = self.dataset.random_iterator.get_next()
        self.pretrain_out = "output"
        self.generator_out = "output"
        self.z_seq_length = "z_seq_length"

    def create_model(self, dropout=False, output_keep_prob=1, forget_bias=1, pretrain=True):
        """Create generator model using lstms for either random text creation or pretraining"""
        with tf.variable_scope("generator_lstm"):
            self.lstmcells = [tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
                              for n_hidden in self.layers]
        if dropout and output_keep_prob < 1:
            cells = [tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=output_keep_prob) for cell in self.lstmcells]
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        else:
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(self.lstmcells)

        self.pretrain_out = self._pretrain_generator(multi_rnn_cell)
        self.generator_out = self._generator(multi_rnn_cell, forget_bias=forget_bias)
        if self.pretrain:
            self.final_output = self.pretrain_out
        else:
            self.final_output = self.generator_out
        print("Generator model built", file=sys.stderr)
        return self.final_output

    def _generator(self, multi_rnn_cell, forget_bias=1):
        with tf.variable_scope("random_generator"):
            states = [tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=forget_bias).zero_state(self.dataset.batch_size,
                                                                                            tf.float32)
                      for n_hidden in self.layers]
            outputs = []
            input_vector = self.x
            for time_step in range(self.dataset.max_seq_len):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, states = multi_rnn_cell(inputs=input_vector, state=states)
                input_vector, _, _ = self.fulconn_layer(cell_output, self.dataset.len_x, activation_func=tf.tanh)
                outputs.append(input_vector)
            # print(outputs)
            final_output = tf.stack(outputs, 1)
        return final_output

    def _pretrain_generator(self, multi_rnn_cell):
        """Create pretrain generator graph"""
        with tf.variable_scope("pretrain_generator"):
            output, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                          inputs=self.p_x,
                                          dtype=tf.float32,
                                          time_major=False,
                                          sequence_length=self.p_seq)
            # get output
            output = tf.reshape(output, shape=[-1, self.layers[-1]])
            final_output, _, _ = self.fulconn_layer(output, self.dataset.len_x, activation_func=tf.tanh)
            # Reshape output back into [batch_size, max_seq_len, len_x]
            final_output = tf.reshape(final_output, shape=[-1, self.dataset.max_seq_len, self.dataset.len_x])
        return final_output

    def create_ops(self):
        """Create operations for training"""
        self.predict = tf.reshape(tf.argmax(self.final_output, 2),
                                  [self.dataset.batch_size, self.dataset.max_seq_len], name="g_predict")

        self.accuracy, self.loss, self.trainable_vars = self._create_pretrain_ops()
        # indexes of most likely
        print("Created Loss functions", file=sys.stderr)
        # create summary info
        self.variable_summaries(self.loss, "generator_loss")
        self.variable_summaries(self.accuracy, "generator_accuracy")
        return True

    def _create_pretrain_ops(self):
        """Create operations for training the pretrain generator"""
        # total right by batch
        sum1 = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(self.pretrain_out, 2), tf.argmax(self.p_y, 2)), dtype=tf.float32),
            axis=1, keep_dims=True)
        # accuracy is number correct / seq len
        accuracy = tf.reduce_mean(tf.divide(sum1, tf.cast(self.p_seq, dtype=tf.float32)), name="g_accuracy")

        # calculate loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.final_output, labels=self.p_y))

        # partition trainable variables
        tvars = tf.trainable_variables()
        trainable_vars = [var for var in tvars if 'generator' in var.name]
        return accuracy, loss, trainable_vars


class TweetDiscriminator(CreateTFNetwork):
    """Create tweet generator using LSTMs"""

    def __init__(self, dataset, log, layers=tuple()):
        self.layers = layers
        self.log = log
        self.lstmcells = "lstmcells"
        self.global_step = tf.get_variable(
            'd_global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        self.final_output = "final output"
        self.trainable_vars = "vars"
        self.len_y = 1

        super(TweetDiscriminator, self).__init__(dataset)
        self.x, self.seq, self.y = self.dataset.iterator.get_next()

    def create_model(self, reuse=False, dropout=False, output_keep_prob=1, forget_bias=1):
        """Create generator model using lstms for either random text creation or pretraining"""
        with tf.variable_scope("discriminator", reuse=reuse):

            self.lstmcells = [tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
                              for n_hidden in self.layers]

            if dropout and output_keep_prob < 1:
                cells = [tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=output_keep_prob) for cell in
                         self.lstmcells]
                multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            else:
                multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(self.lstmcells)

                output, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                              inputs=self.x,
                                              dtype=tf.float32,
                                              time_major=False,
                                              sequence_length=self.seq)

                batch_size = tf.shape(output)[0]
                last_outputs = tf.gather_nd(output, tf.stack([tf.range(batch_size), self.seq - 1], axis=1))

                with tf.variable_scope("final_full_conn_layer", reuse=tf.AUTO_REUSE):
                    self.final_output, weights, bias = self.fulconn_layer(input_data=last_outputs,
                                                                          output_dim=self.len_y)
        print("Discriminator Model Built", file=sys.stderr)

        return self.final_output

    def create_ops(self):
        """Create losses, accuracy and other operations """

        # discriminator accuracy
        # d_accuracy = tf.reduce_mean(tf.cast(tf.equal(Dx, place_Y), tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.sign(tf.nn.relu(self.final_output)), self.y),
                                               tf.float32), name="pretrain_g_accuracy")

        # calculate loss

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.final_output, labels=self.y))
        print("Created Loss function", file=sys.stderr)

        # create summary info
        self.variable_summaries(self.loss, "discriminator_loss")
        self.variable_summaries(self.accuracy, "discriminator_accuracy")

        # partition trainable variables
        tvars = tf.trainable_variables()
        self.trainable_vars = [var for var in tvars if "discriminator" in var.name]
        return True


class TextGan(CreateTFNetwork):
    """Create tweet generator using LSTMs"""

    def __init__(self, discriminator, generator, log):
        self.discriminator = discriminator
        self.generator = generator
        self.log = log
        super(TextGan, self).__init__(self.generator.dataset)
        self.use_generator = tf.placeholder(tf.bool, shape=None)
        self.z_seq_length = "generated seq len"
        self.passed_sentences = "passed generated sentences"
        self.d_vars = "disciminator vars"
        self.g_vars = "generator vars"
        self.generator_summary = []
        self.discriminator_summary_fake = []
        self.discriminator_summary_real = []
        self.g_accuracy = "generator accuracy"
        self.d_accuracy = "discriminator accuracy"
        self.g_loss = "generator loss"
        self.d_loss = "discriminator loss"

    def create_model(self):
        """Create generator and discriminator models"""
        with tf.device('/gpu:0'):
            self.generator.create_model(pretrain=False)

        def index1d(t):
            """Get index of first appearance of specific character"""
            index = tf.cast(tf.reduce_min(tf.where(tf.equal(self.generator.stop_char_index, t))), dtype=tf.int32)
            # return index
            return tf.cond(index < 0, lambda: tf.cast(self.generator.max_seq_len_tensor, dtype=tf.int32),
                           lambda: tf.cast(tf.add(index, 1), dtype=tf.int32))

        # get character indexes for all sequences
        gen_char_index = tf.argmax(self.generator.final_output, axis=2)
        # length of the sequence for the generator network based on termination character
        self.z_seq_length = tf.map_fn(index1d, gen_char_index, dtype=tf.int32, back_prop=False)

        self.discriminator.x = tf.cond(self.use_generator, lambda: self.generator.final_output,
                                       lambda: self.discriminator.x, )
        self.discriminator.seq = tf.cond(self.use_generator, lambda: self.z_seq_length, lambda: self.discriminator.seq)
        # self.discriminator.y = tf.cond(self.use_generator, lambda: self.generator.y, lambda: self.discriminator.y)
        with tf.device('/gpu:1'):
            self.discriminator.create_model()

    def create_ops(self):
        self.predict = tf.reshape(tf.argmax(self.generator.final_output, 2),
                                  [self.generator.dataset.batch_size, 1, self.generator.dataset.max_seq_len],
                                  name="g_predict")

        self.g_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.sign(self.discriminator.final_output),
                                                          tf.ones_like(self.discriminator.final_output)), tf.float32))

        # discriminator accuracy
        d_accuracy_real = tf.reduce_mean(tf.cast(tf.equal(tf.sign(self.discriminator.final_output),
                                                          tf.ones_like(self.discriminator.final_output)),
                                                 tf.float32))
        d_accuracy_fake = tf.reduce_mean(tf.cast(tf.equal(tf.nn.relu(self.discriminator.final_output),
                                                          tf.zeros_like(self.discriminator.final_output)),
                                                 tf.float32))

        self.d_accuracy = tf.cond(self.use_generator, lambda: d_accuracy_fake,
                                  lambda: d_accuracy_real)

        # sentences that passed the discriminator
        indices = tf.where(tf.equal(tf.sign(self.discriminator.final_output),
                                    tf.ones_like(self.discriminator.final_output)))
        self.passed_sentences = tf.gather_nd(self.predict, indices)

        # calculate loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminator.final_output,
                                                                             labels=tf.ones_like(
                                                                                 self.discriminator.final_output)))
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminator.final_output,
                                                                             labels=tf.ones_like(
                                                                                 self.discriminator.final_output)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminator.final_output,
                                                                             labels=tf.zeros_like(
                                                                                 self.discriminator.final_output)))
        self.d_loss = tf.cond(self.use_generator, lambda: d_loss_fake,
                              lambda: d_loss_real)

        d_summaries_real = []
        g_summaries = []
        d_summaries_fake = []
        with tf.name_scope("d_loss_real_tweets"):
            mean = tf.reduce_mean(self.d_loss)
            summary = tf.summary.scalar('mean', mean)
            d_summaries_real.append(summary)
        with tf.name_scope("d_acc_real_tweets"):
            mean = tf.reduce_mean(self.d_accuracy)
            summary = tf.summary.scalar('mean', mean)
            d_summaries_real.append(summary)
        with tf.name_scope("d_loss_fake_tweets"):
            mean = tf.reduce_mean(self.d_loss)
            summary = tf.summary.scalar('mean', mean)
            d_summaries_fake.append(summary)
        with tf.name_scope("d_acc_fake_tweets"):
            mean = tf.reduce_mean(self.d_accuracy)
            summary = tf.summary.scalar('mean', mean)
            d_summaries_fake.append(summary)
        with tf.name_scope("generator_loss"):
            mean = tf.reduce_mean(self.d_loss)
            summary = tf.summary.scalar('mean', mean)
            g_summaries.append(summary)
        with tf.name_scope("generator_accuracy"):
            mean = tf.reduce_mean(self.g_accuracy)
            summary = tf.summary.scalar('mean', mean)
            g_summaries.append(summary)

        self.generator_summary = tf.summary.merge(g_summaries)
        self.discriminator_summary_real = tf.summary.merge(d_summaries_real)
        self.discriminator_summary_fake = tf.summary.merge(d_summaries_fake)

        # partition trainable variables
        tvars = tf.trainable_variables()
        self.d_vars = [var for var in tvars if 'discriminator' in var.name]
        self.g_vars = [var for var in tvars if 'generator' in var.name]
        print("Created Loss functions", file=sys.stderr)


if __name__ == "__main__":
    print("This is a library file. Nothing to execute")
