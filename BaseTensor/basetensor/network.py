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


class CreateTFNetwork(ABC):
    def __init__(self, dataset):
        assert isinstance(dataset, CreateTFDataset)
        self.dataset = dataset
        self.train_op = "train ops"
        self.inference_op = "inference ops"
        super(CreateTFNetwork, self).__init__()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_train_ops(self):
        pass

    @abstractmethod
    def create_inference_op(self):
        pass


if __name__ == "__main__":
    print("This is a library file. Nothing to execute")
    raise SystemExit
