#!/usr/bin/env python3
"""Abstract class for dataset and iterator creation using tensorflow"""
########################################################################
# File: dataset.py
#  executable: dataset.py

# Author: Andrew Bailey
# History: 12/07/17 Created
########################################################################

from abc import ABC, abstractmethod


class CreateTFDataset(ABC):
    def __init__(self):
        self.iterator = "iterator"
        super(CreateTFDataset, self).__init__()

    @abstractmethod
    def create_dataset(self):
        pass

    @abstractmethod
    def create_iterator(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def load_training_data(self):
        pass

    @abstractmethod
    def process_graph_output(self):
        pass

    @abstractmethod
    def load_inference_data(self):
        pass


if __name__ == "__main__":
    print("This is a library file. Nothing to execute")
    raise SystemExit
