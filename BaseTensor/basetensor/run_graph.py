#!/usr/bin/env python3
"""Abstract class for running models"""
########################################################################
# File: run_graph.py
#  executable: run_graph.py

# Author: Andrew Bailey
# History: 12/08/17 Created
########################################################################

from abc import ABC, abstractmethod
from basetensor.network import CreateTFNetwork
import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline


class BasicTFTraining(ABC):
    """Boilerplate abstract class for running tensorflow models"""
    def __init__(self, model):
        assert isinstance(model, CreateTFNetwork)
        self.network = model
        super(BasicTFTraining, self).__init__()

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def run_model(self):
        pass

    @abstractmethod
    def test_model(self):
        pass


    # def profile_training(self, sess, writer, run_metadata, run_options):
    #     """Expensive profile step so we can track speed of operations of the model"""
    #     _, summary, global_step = sess.run(
    #         [self.train_op, self.summaries, self.global_step],
    #         run_metadata=run_metadata, options=run_options)
    #     # add summary statistics
    #     writer.add_summary(summary, global_step)
    #     writer.add_run_metadata(run_metadata, "step{}_train".format(global_step))
    #     if self.args.save_trace:
    #         self.chrome_trace(run_metadata, self.args.trace_name)
    #
    # @staticmethod
    # def chrome_trace(metadata_proto, f_name):
    #     """Save a chrome trace json file.
    #     To view json vile go to - chrome://tracing/
    #     """
    #     time_line = timeline.Timeline(metadata_proto.step_stats)
    #     ctf = time_line.generate_chrome_trace_format()
    #     with open(f_name, 'w') as file1:
    #         file1.write(ctf)
    #
    #
    # def test_time(self):
    #     """Return true if it is time to save the model"""
    #     delta = (datetime.now() - self.start).total_seconds()
    #     if delta > self.args.save_model:
    #         self.start = datetime.now()
    #         return True
    #     return False
    #
    # def get_model_files(self, *files):
    #     """Collect neccessary model files for upload"""
    #     file_list = [self.model_path + ".data-00000-of-00001", self.model_path + ".index"]
    #     for file1 in files:
    #         file_list.append(file1)
    #     return file_list
    #
    #
    # def average_gradients(tower_grads):
    #     """Calculate the average gradient for each shared variable across all towers.
    #     Note that this function provides a synchronization point across all towers.
    #     Args:
    #       tower_grads: List of lists of (gradient, variable) tuples. The outer list
    #         is over individual gradients. The inner list is over the gradient
    #         calculation for each tower.
    #     Returns:
    #        List of pairs of (gradient, variable) where the gradient has been averaged
    #        across all towers.
    #     """
    #     average_grads = []
    #     # # print(tower_grads)
    #     for grad_and_vars in zip(*tower_grads):
    #         # Note that each grad_and_vars looks like the following:
    #         #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    #         # print(grad_and_vars)
    #         grads = []
    #         for g, v in grad_and_vars:
    #             # print(g)
    #             # print(v)
    #             # print("Another gradient and variable")
    #             # Add 0 dimension to the gradients to represent the tower.
    #             expanded_g = tf.expand_dims(g, 0)
    #
    #             # Append on a 'tower' dimension which we will average over below.
    #             grads.append(expanded_g)
    #
    #         # Average over the 'tower' dimension.
    #         grad = tf.concat(axis=0, values=grads)
    #         grad = tf.reduce_mean(grad, 0)
    #         #
    #         # # Keep in mind that the Variables are redundant because they are shared
    #         # # across towers. So .. we will just return the first tower's pointer to
    #         # # the Variable.
    #         v = grad_and_vars[0][1]
    #         grad_and_var = (grad, v)
    #         average_grads.append(grad_and_var)
    #     return average_grads
    #
    #
    # def test_for_nvidia_gpu(num_gpu):
    #     assert type(num_gpu) is int, "num_gpu option must be integer"
    #     if num_gpu == 0:
    #         return False
    #     else:
    #         try:
    #             utilization = re.findall(r"Utilization.*?Gpu.*?(\d+).*?Memory.*?(\d+)",
    #                                      subprocess.check_output(["nvidia-smi", "-q"]),
    #                                      flags=re.MULTILINE | re.DOTALL)
    #             indices = [i for i, x in enumerate(utilization) if x == ('0', '0')]
    #             assert len(indices) >= num_gpu, "Only {0} GPU's are available, change num_gpu parameter to {0}".format(
    #                 len(indices))
    #             return indices[:num_gpu]
    #         except OSError:
    #             log.info("No GPU's found. Using CPU.")
    #             return False
    #
    #
    # def optimistic_restore(session, save_file):
    #     """ Implementation from: https://github.com/tensorflow/tensorflow/issues/312 """
    #     print('Restoring model from:', save_file)
    #     reader = tf.train.NewCheckpointReader(save_file)
    #     saved_shapes = reader.get_variable_to_shape_map()
    #     var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
    #                         if var.name.split(':')[0] in saved_shapes])
    #     restore_vars = []
    #     name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    #     with tf.variable_scope('', reuse=True):
    #         for var_name, saved_var_name in var_names:
    #             curr_var = name2var[saved_var_name]
    #             var_shape = curr_var.get_shape().as_list()
    #             if var_shape == saved_shapes[saved_var_name]:
    #                 restore_vars.append(curr_var)
    #     saver = tf.train.Saver(restore_vars)
    #     saver.restore(session, save_file)
