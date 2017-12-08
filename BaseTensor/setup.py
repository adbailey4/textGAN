#!/usr/bin/env python3
"""Create setup script for pip installation of BaseTensor"""
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
        name="basetensor",
        version='0.0.1',
        description='Abstract classes for tensorflow and some utility functions',
        url='https://github.com/adbailey4/BaseTensor',
        author='Andrew Bailey',
        author_email='andbaile@ucsc.com',
        packages=['basetensor'],
        install_requires=["tensorflow >= 1.4.0"],
        zip_safe=True
    )

    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
