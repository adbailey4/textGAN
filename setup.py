#!/usr/bin/env python3
"""Create setup script for pip installation of textGAN"""
########################################################################
# File: setup.py
#  executable: setup.py
#
# Author: Andrew Bailey
# History: 12/07/17 Created
########################################################################

import sys
from timeit import default_timer as timer
from setuptools import setup


def main():
    """Main docstring"""
    start = timer()
    setup(
        name="textGAN",
        version='0.0.1',
        description='Tensorflow implementation of a Generative Adversarial Network',
        url='https://github.com/adbailey4/textGAN',
        author='Andrew Bailey',
        author_email='andbaile@ucsc.com',
        packages=['textgan'],
        install_requires=["tensorflow>=1.4.0",
                          "unicodecsv==0.14.1",
                          "unidecode==0.4.21"],
        zip_safe=True
    )

    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
