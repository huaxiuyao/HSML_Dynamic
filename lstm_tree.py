import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell
from tensorflow.contrib.learn import KMeansClustering

import numpy as np
from tensorflow.python.platform import flags
import sys
FLAGS = flags.FLAGS
import ipdb


class TreeLSTM(object):
    def __init__(self, tree_hidden_dim, input_dim, cluster_layer_0, cluster_layer_1):
        self.input_dim = input_dim
        self.tree_hidden_dim = tree_hidden_dim
        self.leaf_weight_i, self.leaf_weight_o, self.leaf_weight_u = [], [], []
        self.leaf_bias_i, self.leaf_bias_o, self.leaf_bias_u = [], [], []

        self.cluster_layer_0 = cluster_layer_0
        self.cluster_layer_1 = cluster_layer_1

        for i in range(self.cluster_layer_0):
            self.leaf_weight_u.append(
                tf.get_variable(name='{}_leaf_weight_u'.format(i), shape=(input_dim, tree_hidden_dim)))
            self.leaf_bias_u.append(tf.get_variable(name='{}_leaf_bias_u'.format(i), shape=(1, tree_hidden_dim)))

        self.no_leaf_weight_i, self.no_leaf_weight_o, self.no_leaf_weight_u, self.no_leaf_weight_f = [], [], [], []
        self.no_leaf_bias_i, self.no_leaf_bias_o, self.no_leaf_bias_u, self.no_leaf_bias_f = [], [], [], []
        for i in range(self.cluster_layer_1):
            if FLAGS.tree_type==1:
                self.no_leaf_weight_i.append(
                    tf.get_variable(name='{}_no_leaf_weight_i'.format(i), shape=(tree_hidden_dim, 1)))
            elif FLAGS.tree_type==2:
                self.no_leaf_weight_i.append(
                    tf.get_variable(name='{}_no_leaf_weight_i'.format(i), shape=(1, tree_hidden_dim)))
            self.no_leaf_weight_u.append(
                tf.get_variable(name='{}_no_leaf_weight_u'.format(i), shape=(tree_hidden_dim, tree_hidden_dim)))

            self.no_leaf_bias_i.append(tf.get_variable(name='{}_no_leaf_bias_i'.format(i), shape=(1, 1)))
            self.no_leaf_bias_u.append(tf.get_variable(name='{}_no_leaf_bias_u'.format(i), shape=(1, tree_hidden_dim)))

        self.root_weight_u = tf.get_variable(name='{}_root_weight_u'.format(i),
                                             shape=(tree_hidden_dim, tree_hidden_dim))

        self.root_bias_u = tf.get_variable(name='{}_root_bias_u'.format(i), shape=(1, tree_hidden_dim))

        self.cluster_center = []
        for i in range(self.cluster_layer_0):
            self.cluster_center.append(tf.get_variable(name='{}_cluster_center'.format(i),
                                                       shape=(1, input_dim)))

    def model(self, inputs):

        if FLAGS.datasource == 'multidataset' or FLAGS.datasource == 'miniimagenet':
            sigma=10.0
        elif FLAGS.datasource in ['sinusoid', 'mixture']:
            sigma=2.0

        for idx in range(self.cluster_layer_0):
            if idx == 0:
                all_value = tf.exp(-tf.reduce_sum(tf.square(inputs - self.cluster_center[idx])) / (2.0*sigma))
            else:
                all_value += tf.exp(-tf.reduce_sum(tf.square(inputs - self.cluster_center[idx])) / (2.0*sigma))

        c_leaf = []
        for idx in range(self.cluster_layer_0):
            assignment_idx = tf.exp(-tf.reduce_sum(tf.square(inputs - self.cluster_center[idx])) / (2.0*sigma)) / all_value
            value_u = tf.tanh(tf.matmul(inputs, self.leaf_weight_u[idx]) + self.leaf_bias_u[idx])
            value_c = assignment_idx * value_u
            c_leaf.append(value_c)

        c_no_leaf = []
        for idx in range(self.cluster_layer_0):
            input_gate = []
            for idx_layer_1 in range(self.cluster_layer_1):
                if FLAGS.tree_type == 1:
                    input_gate.append(
                        tf.matmul(c_leaf[idx], self.no_leaf_weight_i[idx_layer_1]) + self.no_leaf_bias_i[idx_layer_1])
                elif FLAGS.tree_type == 2:
                    input_gate.append(
                        -(tf.reduce_sum(tf.square(c_leaf[idx] - self.no_leaf_weight_i[idx_layer_1]), keepdims=True) +
                          self.no_leaf_bias_i[idx_layer_1]) / (
                            2.0))
            input_gate = tf.nn.softmax(tf.concat(input_gate, axis=0), axis=0)
            c_no_leaf_temp = []
            for idx_layer_1 in range(self.cluster_layer_1):
                no_leaf_value_u = tf.tanh(
                    tf.matmul(c_leaf[idx], self.no_leaf_weight_u[idx_layer_1]) + self.no_leaf_bias_u[idx_layer_1])
                c_no_leaf_temp.append(input_gate[idx_layer_1] * no_leaf_value_u)
            c_no_leaf.append(tf.concat(c_no_leaf_temp, axis=0))

        c_no_leaf = tf.stack(c_no_leaf, axis=0)
        c_no_leaf = tf.transpose(c_no_leaf, perm=[1, 0, 2])
        c_no_leaf = tf.reduce_sum(c_no_leaf, axis=1, keepdims=True)

        root_c = []

        for idx in range(self.cluster_layer_1):
            root_c.append(tf.tanh(tf.matmul(c_no_leaf[idx], self.root_weight_u) + self.root_bias_u))

        root_c = tf.reduce_sum(tf.concat(root_c, axis=0), axis=0, keepdims=True)

        return root_c, root_c
