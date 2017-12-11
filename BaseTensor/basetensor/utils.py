#!/usr/bin/env python3
"""Tensorflow utility functions"""
########################################################################
# File: utils.py
#  executable: utils.py

# Author: Andrew Bailey
# History: 12/07/17 Created
########################################################################
from datetime import datetime
import os
import json
import logging as log
import tensorflow as tf

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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


def load_json(path):
    """Load a json file and make sure that path exists"""
    path = os.path.abspath(path)
    assert os.path.isfile(path), "Json file does not exist: {}".format(path)
    with open(path) as json_file:
        args = json.load(json_file)
    return args


def save_json(dict1, path):
    """Save a python object as a json file"""
    path = os.path.abspath(path)
    with open(path, 'w') as outfile:
        json.dump(dict1, outfile)
    assert os.path.isfile(path)
    return path


def time_it(func, *args):
    """Very basic timing function
    :param func: callable function
    :param args: arguments to pass to function
    :return: object returned from function, time to complete
    """
    start = datetime.now()
    assert callable(func), "Function is not callable"
    something = func(*args)
    end = datetime.now()
    return something, end-start


def debug(verbose=False):
    """Method for setting log statements with verbose or not verbose"""
    assert type(verbose) is bool, "Verbose needs to be a boolean"
    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")
        log.info("This should not print.")


def optimistic_restore(session, save_file):
    """ Implementation from: https://github.com/tensorflow/tensorflow/issues/312 """
    print('Restoring model from:', save_file)
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)



if __name__ == "__main__":
    print("This is a library file. Nothing to execute")
    raise SystemExit
