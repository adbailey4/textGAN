#!/usr/bin/env python3
"""Tensorflow utility functions"""
########################################################################
# File: utils.py
#  executable: utils.py

# Author: Andrew Bailey
# History: 12/07/17 Created
########################################################################
from datetime import datetime


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


if __name__ == "__main__":
    print("This is a library file. Nothing to execute")
    raise SystemExit
