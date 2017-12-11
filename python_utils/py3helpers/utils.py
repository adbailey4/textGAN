#!/usr/bin/env python
"""Utility functions and classes for python"""
########################################################################
# File: utils.py
#  executable: utils.py
#
# Author: Andrew Bailey
# History: 12/09/17 Created
########################################################################

import logging


def create_logger(file_name, verbose=False, debug=False):
    """Create a logger instance which will write to file path"""
    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    if verbose:
        level = logging.INFO
    # format input
    name = " "
    root_logger = logging.getLogger(name)
    root_logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)s] [%(levelname)s]  %(message)s")

    file_handler = logging.FileHandler("{}.log".format(file_name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(level)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(name)


if __name__ == '__main__':
    print("This is a library of python functions")

