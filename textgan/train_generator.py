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
import argparse
from basetensor.utils import DotDict, list_dir, load_json, save_json
from textgan.tweet_datasets import TweetGeneratorDataset
from textgan.models import TweetGenerator
from py3helpers.utils import create_logger
from textgan.train_textgan import GanTraining, load_gan_params

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
        self.parser.add_argument('-g', '--generator_config',
                                 help='path to json generator config')
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

def main():
    ##################################
    pretrain_generator = True

    command_line = CommandLine()
    gen_config_path = command_line.args["generator_config"]
    assert gen_config_path, "Must specify Generator config"

    params = load_gan_params(gen_config_path, name="generator")
    print("Model Path: {}".format(params.model_path), file=sys.stderr)

    # gan_config_path = command_line.args["gan_config"]
    # dis_config_path = command_line.args["discriminator_config"]
    log1 = create_logger(params.model_path, verbose=command_line.args["verbose"],
                         debug=command_line.args["debug"])

    file_list = list_dir(params.twitter_data_path, ext="csv")

    tweets = TweetGeneratorDataset(log=log1)
    tweets.load_training_data(file_list, reduction_level=params.reduction_level, end_tweet_char=params.end_tweet_char)
    tweets.create_dataset(batch_size=params.batch_size, n_epochs=1, pretrain=True)
    tweets.create_iterator()
    tweets.test()

    gen_model = TweetGenerator(tweets, params.layers, log=log1)
    gen_model.create_model()
    gen_model.create_ops()
    gan_training = GanTraining(gen_model, gen_model, log=log1)
    gan_training.pretrain_generator(params)


if __name__ == '__main__':
    main()
