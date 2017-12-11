#!/usr/bin/env python
"""Create inputs to a tensorflow graph using tf operations and queues"""
########################################################################
# File: queue.py
#  executable: queue.py

# Author: Andrew Bailey
# History: 06/05/17 Created
########################################################################

from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
import csv
from unidecode import unidecode

from basetensor.abstract import CreateTFDataset


# do you want to live life on the edge?  then leave this line uncommented, you badass!
# import warnings
#
# warnings.filterwarnings("ignore")


class TweetGeneratorDataset(CreateTFDataset):
    """Create dataset and iterator for the generation of tweet data"""

    def __init__(self, log):
        self.max_seq_len = 0
        self.all_tweets = []
        self.all_seq_len = []
        self.len_x = 0
        self.max_seq_len = 0
        # create translation dictionaries
        self.ix_to_char = dict()
        self.char_to_ix = dict()
        self.batch_size = 0
        self.len_y = 1
        self.end_tweet_char = u'\u26D4'
        self.dataset = tf.data.Dataset
        self.pretrain_iterator = tf.data.Iterator
        self.random_iterator = tf.data.Iterator
        self.pretrain = bool()
        self.random_dataset = tf.data.Dataset
        self.handle = tf.placeholder(tf.string, shape=[])
        self.log = log
        super(CreateTFDataset, self).__init__()

    def create_iterator(self):
        self.pretrain_iterator = self.dataset.make_initializable_iterator()
        self.random_iterator = self.random_dataset.make_initializable_iterator()
        return True

    def test(self):
        # pretraining data
        in_1, seq, y = self.pretrain_iterator.get_next()
        with tf.Session() as sess:
            sess.run(self.pretrain_iterator.initializer)
            test1 = sess.run([in_1])
            test2 = sess.run([seq])
            test3 = sess.run([y])
        # not pretraining data
        in_1= self.random_iterator.get_next()
        with tf.Session() as sess:
            sess.run(self.random_iterator.initializer)
            test1 = sess.run([in_1])

        print("Dataset Creation Complete", file=sys.stderr)

    def load_training_data(self, file_list, reduction_level, end_tweet_char=u'\u26D4'):
        """Read in tweet data from csv file"""
        self.end_tweet_char = end_tweet_char
        self.len_x, self.max_seq_len, self.ix_to_char, self.char_to_ix, self.all_tweets, self.all_seq_len = \
            load_tweet_data(file_list, reduction_level, end_tweet_char=end_tweet_char)
        return self.len_x, self.max_seq_len, self.ix_to_char, self.char_to_ix, self.all_tweets, self.all_seq_len

    def create_dataset(self, batch_size, n_epochs=1, shuffle_buffer_size=10000, pretrain=True):
        """Create dataset object for tweet generator training
        :param batch_size: defines size of batchs
        :param n_epochs: max number of iterations through data
        :param shuffle_buffer_size: size of buffer for shuffling data
        :return: tf.dataset object
        """
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.dataset = self._create_pretrain_dataset(shuffle_buffer_size=shuffle_buffer_size)
        self.random_dataset = self._create_random_dataset(shuffle_buffer_size=shuffle_buffer_size)

    def _create_pretrain_dataset(self, n_epochs=1, shuffle_buffer_size=10000):
        """Create dataset object for tweet generator training
        :param batch_size: defines size of batchs
        :param n_epochs: max number of iterations through data
        :param shuffle_buffer_size: size of buffer for shuffling data
        :return: tf.dataset object
        """
        # creates dataset from a generator
        dataset = tf.data.Dataset.from_generator(
            self._create_pretrain_data_generator, (tf.float32, tf.int32, tf.float32),
            (tf.TensorShape([self.max_seq_len, self.len_x]),
             tf.TensorShape(None),
             tf.TensorShape([self.max_seq_len, self.len_x])))

        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(n_epochs)
        self.dataset = dataset.prefetch(buffer_size=4)
        return self.dataset

    def _create_random_dataset(self, n_epochs=1, shuffle_buffer_size=10000):
        """Create dataset object for tweet generator training
          :param batch_size: defines size of batchs
          :param n_epochs: max number of iterations through data
          :param shuffle_buffer_size: size of buffer for shuffling data
          :return: tf.dataset object
        """
        # creates dataset from a generator
        dataset = tf.data.Dataset.from_generator(
            self._create_random_data_generator, tf.float32,
            tf.TensorShape([self.len_x]))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(n_epochs)
        dataset = dataset.prefetch(buffer_size=4)

        return dataset

    def _create_pretrain_data_generator(self):
        """Creates a generator of batches for the tf.dataset object"""
        while True:
            for tweet in self.all_tweets:
                vector_tweet = np.zeros([self.max_seq_len, self.len_x])
                label_tweet = np.zeros([self.max_seq_len, self.len_x])
                for indx, char in enumerate(tweet):
                    vector_tweet[indx, self.char_to_ix[char]] = 1
                    if indx == len(tweet) - 1:
                        label_tweet[indx, self.char_to_ix[self.end_tweet_char]] = 1
                    else:
                        label_tweet[indx, self.char_to_ix[tweet[indx + 1]]] = 1
                        # add tweet ending character to tweet

                yield np.asarray(vector_tweet), indx + 1, np.asarray(label_tweet)

    def _create_random_data_generator(self):
        """Creates a generator of batches for the tf.dataset object"""
        while True:
            z_batch = np.random.normal(0, 1, size=[self.len_x])
            yield z_batch

    def process_graph_output(self):
        pass

    def load_inference_data(self):
        pass


class TweetDiscriminatorDataset(CreateTFDataset):
    """Create dataset and iterator for the generation of tweet data"""

    def __init__(self, log):
        self.log = log
        self.max_seq_len = 0
        self.all_tweets = []
        self.all_seq_len = []
        self.len_x = 0
        self.max_seq_len = 0
        # create translation dictionaries
        self.ix_to_char = dict()
        self.char_to_ix = dict()
        self.batch_size = 0
        self.len_y = 1
        self.end_tweet_char = u'\u26D4'
        self.dataset = tf.data.Dataset
        self.iterator = tf.data.Iterator
        self.swap_prob = 0
        self.fake_dataset = tf.data.Dataset
        self.real_dataset = tf.data.Dataset
        self.iterator = tf.data.Iterator
        self.real_iterator = tf.data.Iterator
        self.fake_iterator = tf.data.Iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        super(CreateTFDataset, self).__init__()

    def create_dataset(self, batch_size, n_epochs, shuffle_buffer_size=1000, pretrain=True, swap_prob=0.1):
        """Create dataset object for tweet generator training
        :param batch_size: defines size of batchs
        :param n_epochs: max number of iterations through data
        :param shuffle_buffer_size: size of buffer for shuffling data
        :return: tf.dataset object
        """
        self.batch_size = batch_size
        self.swap_prob = swap_prob
        # creates dataset from a generator
        fake_dataset = tf.data.Dataset.from_generator(
            self._fake_tweet_generator, (tf.float32, tf.int32, tf.float32),
            (tf.TensorShape([self.max_seq_len, self.len_x]),
             tf.TensorShape(None),
             tf.TensorShape(None)))

        fake_dataset = fake_dataset.shuffle(buffer_size=shuffle_buffer_size)
        fake_dataset = fake_dataset.batch(self.batch_size)
        fake_dataset = fake_dataset.repeat(n_epochs)
        self.fake_dataset = fake_dataset.prefetch(buffer_size=4)

        real_dataset = tf.data.Dataset.from_generator(self._real_tweet_generator,
                                                      (tf.float32, tf.int32, tf.float32),
                                                      (tf.TensorShape([self.max_seq_len, self.len_x]),
                                                       tf.TensorShape(None),
                                                       tf.TensorShape(None)))
        real_dataset = real_dataset.shuffle(buffer_size=shuffle_buffer_size)
        real_dataset = real_dataset.batch(self.batch_size)
        real_dataset = real_dataset.repeat(n_epochs)
        self.real_dataset = real_dataset.prefetch(buffer_size=4)

        return True

    def create_iterator(self):
        self.real_iterator = self.real_dataset.make_initializable_iterator()
        self.fake_iterator = self.fake_dataset.make_initializable_iterator()

        self.iterator = tf.data.Iterator.from_string_handle(self.handle, output_types=self.real_dataset.output_types,
                                                            output_shapes=self.real_dataset.output_shapes)

        return self.iterator

    def test(self):
        in_1, seq, y = self.iterator.get_next()
        with tf.Session() as sess:
            fake_iterator_handle = sess.run(self.fake_iterator.string_handle())
            real_iterator_handle = sess.run(self.real_iterator.string_handle())
            sess.run(self.real_iterator.initializer)
            sess.run(self.fake_iterator.initializer)

            test1 = sess.run([in_1], feed_dict={self.handle: fake_iterator_handle})
            test2 = sess.run([seq], feed_dict={self.handle: fake_iterator_handle})
            test3 = sess.run([y], feed_dict={self.handle: fake_iterator_handle})
            test1 = sess.run([in_1], feed_dict={self.handle: real_iterator_handle})
            test2 = sess.run([seq], feed_dict={self.handle: real_iterator_handle})
            test3 = sess.run([y], feed_dict={self.handle: real_iterator_handle})

        print("Dataset Creation Complete", file=sys.stderr)

    def load_training_data(self, file_list, reduction_level, end_tweet_char=u'\u26D4'):
        """Read in tweet data from csv file"""
        self.end_tweet_char = end_tweet_char
        self.len_x, self.max_seq_len, self.ix_to_char, self.char_to_ix, self.all_tweets, self.all_seq_len = \
            load_tweet_data(file_list, reduction_level, end_tweet_char=end_tweet_char)
        return self.len_x, self.max_seq_len, self.ix_to_char, self.char_to_ix, self.all_tweets, self.all_seq_len

    def _real_tweet_generator(self):
        """Read in data as needed by the batch"""
        while True:
            for tweet in self.all_tweets:
                vector_tweet = np.zeros([self.max_seq_len, self.len_x])
                for indx, char in enumerate(tweet):
                    vector_tweet[indx, self.char_to_ix[char]] = 1
                # add tweet ending character to tweet
                vector_tweet[indx + 1, self.char_to_ix[self.end_tweet_char]] = 1

                yield np.asarray(vector_tweet), indx + 2, 1

    def _fake_tweet_generator(self):
        """Read in data for pretraining the generator"""
        while True:
            for tweet in self.all_tweets:
                fake_tweet = self._create_fake_tweet(tweet)
                vector_tweet = np.zeros([self.max_seq_len, self.len_x])
                for indx, char in enumerate(fake_tweet):
                    vector_tweet[indx, self.char_to_ix[char]] = 1
                # add tweet ending character to tweet
                vector_tweet[indx + 1, self.char_to_ix[self.end_tweet_char]] = 1
                yield np.asarray(vector_tweet), indx + 2, 0

    def _create_fake_tweet(self, tweet):
        """Swap words in tweet"""
        words = tweet.split()
        if len(words) > 1:
            n_swaps = int((len(words) * self.swap_prob) / 2)
            # print(len(words), prob, n_swaps)

            index = np.random.choice(range(0, len(words) - 1, 2), n_swaps, replace=False)
            for i in index:
                tmp_word = words[i]
                words[i] = words[i + 1]
                words[i + 1] = tmp_word
            return ' '.join(words)
        else:
            return tweet

    def process_graph_output(self):
        pass

    def load_inference_data(self):
        pass


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


# this prints the following error:
#   "RuntimeWarning: Surrogate character u'\udc23' will be ignored. You might be using a narrow Python build"
def reduce_unicode_characters(unicode_str, reduction_level=RL_MED):
    """reduces all characters from unicode to less character representation depending on reduction_level

    :param unicode_str: unicode string to reduce number of characters
    :param reduction_level: level of reduction
    :return: unicode string
    """
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


def read_tweet_csv(filename, reduction_level):
    """
    Read in tweet collection of "HillaryClinton" and "realDonaldTrump" (or "none in case of test data).
    Other twitter handles would break this function.
    :param filename: File to read tweet data from.
    :param reduction_level: integer between 0-3 denoting character reduction level
    :return: list with handles and list with tweets; in order of file appearance.
    """
    handles = []
    tweets = []
    with open(filename, 'r', encoding='utf-8') as fileH:
        # goodHandles = ["HillaryClinton", "realDonaldTrump", "none"]
        r = csv.reader(fileH)
        for row in r:
            handles.append(row[0].encode("utf-8"))
            tweets.append(reduce_unicode_characters(row[1], reduction_level=reduction_level))

    return handles, tweets


def load_tweet_data(file_list, reduction_level, end_tweet_char=u'\u26D4'):
    """Read in tweet data from csv file
    source: https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/
    """
    all_tweets = []
    all_seq_len = []
    # collect all tweets from csv files
    for tweet_file in file_list:
        _, tweets = read_tweet_csv(tweet_file, reduction_level=reduction_level)
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


if __name__ == "__main__":
    print("This is a library file. Nothing to execute")


