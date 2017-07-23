#!/usr/bin/env python3
#-*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import math
from six.moves import xrange
import tensorflow as tf
import numbers
import os
import iter_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.layers.python.layers import utils

epsilon = 1e-3

def get_optimizer(opt):
    if opt == "amd":
        optfn = tf.train.AdamOptimizer
    else:
        optfn = tf.train.GradientDescentOptimizer
    return optfn

def prelu(_x):
    with tf.variable_scope("prelu"):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
        pos = tf.nn.relu6(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

def batch_norm_wrapper(inputs, is_training, decay = 0.99):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

class FilterRNN(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, W, V, input_size=None, activation=tanh):
        if input_size is not None:
            print("%s: The input_size parameter is deprecated." % self)
        self._num_units = num_units * 20
        self._activation = activation
        self._W = W
        self._V = V

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            W = tf.get_variable(name="W", shape=self._W.shape, initializer = tf.constant_initializer(self._W), trainable= False)
            V = tf.get_variable(name="V", shape=self._V.shape, initializer = tf.constant_initializer(self._V), trainable= False)
            new_h = tf.matmul(inputs, W) + tf.matmul(state, V)
            return new_h, new_h


class attenCNN(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, batch_size, input_size=None, activation=tanh, trainable = True):
        if input_size is not None:
            print("%s: The input_size parameter is deprecated." % self)
        self._num_units = num_units
        self._activation = activation
        self._batch_size = batch_size
        self._trainable = trainable
        self._W_list = []
        self._B_list = []

    @property
    def state_size(self):
        return self._num_units * 4

    @property
    def output_size(self):
        return self._num_units * 4

    def dump(self):
        return [self._W_omega, self._b_omega, self._u_omega, self._W_list, self._B_list]

    # Convolution model
    def conv2d(self, x, W, b, stride=1):
        # Conv2D wrapper, with bias and sigmoid activation
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # Max-pooling
    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1],
                              padding='SAME')

    # Average pooling
    def avgpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.avg_pool(x, ksize=[1, k, 1, 1], strides=[1, k, k, 1],
                             padding='SAME')

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            reshaped = tf.reshape(inputs, [self._batch_size, 20, self._num_units])

            W_omega = tf.get_variable(name="W_omega", shape=iter_utils.loadWData("att_w_omega.npy").shape, initializer = tf.constant_initializer(iter_utils.loadWData("att_w_omega.npy")), trainable= False)
            b_omega = tf.get_variable(name="b_omega", shape=iter_utils.loadWData("att_b_omega.npy").shape, initializer = tf.constant_initializer(iter_utils.loadWData("att_b_omega.npy")), trainable= False)
            u_omega = tf.get_variable(name="u_omega", shape=iter_utils.loadWData("att_u_omega.npy").shape, initializer = tf.constant_initializer(iter_utils.loadWData("att_u_omega.npy")), trainable= False)
#            W_omega = tf.get_variable("W_omega",[self._num_units, self._num_units], trainable = self._trainable)
#            b_omega = tf.get_variable("b_omega",[self._num_units], trainable = self._trainable)
#            u_omega = tf.get_variable("u_omega",[self._num_units], trainable = self._trainable)

            self._W_omega = W_omega
            self._b_omega = b_omega
            self._u_omega = u_omega

            v = tf.tanh(tf.matmul(tf.reshape(reshaped, [-1, self._num_units]), W_omega) + tf.reshape(b_omega, [1, -1]))
            vu = tf.matmul(v, tf.reshape(u_omega, [-1,1]))
            exps = tf.reshape(tf.exp(vu), [-1, 20])
            alphas = exps/(tf.reshape(tf.reduce_sum(exps, 1), [-1,1]) + 1e-9)
            att_output = reshaped * tf.reshape(alphas, [-1, 20, 1])

            input_4D = tf.expand_dims(att_output, -1)    #yields [batch x r x d x 1]
            filter_sizes = [2,3,4,5]
            pooled_outputs = []

            for i, filter_size in enumerate(filter_sizes):
                with vs.variable_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self._num_units, 1, self._num_units]
                    #W = tf.get_variable('W', filter_shape, trainable = self._trainable)
                    W = tf.get_variable(name="W", shape=iter_utils.loadWData("att_w_%d.npy"%i).shape, initializer = tf.constant_initializer(iter_utils.loadWData("att_w_%d.npy"%i)), trainable= False)
                    B = tf.get_variable(name="B", shape=iter_utils.loadWData("mask_b_%d.npy"%i).shape, initializer = tf.constant_initializer(iter_utils.loadWData("mask_b_%d.npy"%i)), trainable= False)
                    #B = tf.get_variable('B', [self._num_units], trainable = self._trainable)
                    if len(self._W_list) != len(filter_sizes):
                        self._W_list.append(W)
                    if len(self._B_list) != len(filter_sizes):
                        self._B_list.append(B)

                    conv = tf.nn.conv2d(
                        input_4D,
                        W,
                        strides = [1,1,1,1],
                        padding = "VALID",
                        name = "conv")
                    with vs.variable_scope(str(i)) as scope:
                        h = prelu(tf.nn.bias_add(conv, B))

                    max_pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, 20 - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="max_pool")

                    avg_pooled = tf.nn.avg_pool(
                        h,
                        ksize=[1, 20 - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="avg_pool")

                    pooled = max_pooled + avg_pooled
                    pooled_outputs.append(pooled)

            num_filters_total = 256 * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, axis = 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            return h_pool_flat, h_pool_flat

class dataTagger(object):
    def __init__(self, batch_size, vocab_size, state_size, att_state_size, sen_state_size, max_gradient_norm, learning_rate,
                 learning_rate_decay_factor, coeff_config, dropout, weight, forward_only = False, optimizer = "adam",
                 word2vec = None, c_word2vec = None, w_mask = None, W = None, V = None):
        self._W = W
        self._V = V
        self._w_mask = w_mask
        self._word2vec = word2vec
        self._c_word2vec = c_word2vec
        self._forward_only = forward_only
        self._batch_size = batch_size
        self._state_size = state_size
        self._att_state_size = att_state_size
        self._sen_state_size = sen_state_size
        self._vocab_size = vocab_size
        self._keep_prob_config = 1 - dropout
        self._coeff_config = coeff_config
        self._learning_rate = tf.Variable(float(learning_rate), trainable = False)
        self._learning_rate_decay_op = self._learning_rate.assign(self._learning_rate * learning_rate_decay_factor)
        self._weight = weight
        self._global_step = tf.Variable(0, trainable=False)
        self._keep_prob = tf.placeholder(tf.float32)
        self._coeff = tf.placeholder(tf.float32)
        self._labelLen = tf.placeholder(tf.int32)
        self._srclen = tf.placeholder(tf.int32, shape = [batch_size])
        self._question_tokens = tf.placeholder(tf.int32, shape = [batch_size, None])
        self._c_srclen = tf.placeholder(tf.int32, shape = [batch_size])
        self._c_question_tokens = tf.placeholder(tf.int32, shape = [batch_size, None])
        self._mark = tf.placeholder(tf.float32, shape = [batch_size, 1999])
        self._labels = tf.placeholder(tf.int32, shape = [batch_size, 1999])

        with tf.variable_scope("dataTagger", initializer = tf.uniform_unit_scaling_initializer()):
            self.setup_embeddings()
#            self.setup_ACM()
#            self.setup_glu()
#            self.setup_encoder()
#            self.setup_c_encoder()
            self.setup_CNN()
            self.setup_c_CNN()
#            self.setup_mark()
            self.setup_decoder()
#            self.setup_decoder1()
#            self.setup_decoder2()
            self.setup_loss()
            self.setup_summary()

        if not forward_only:
            params = tf.trainable_variables()
            gradients = tf.gradients(self._loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            opt = tf.train.AdamOptimizer(learning_rate = self._learning_rate)
            train_step = opt.minimize(self._loss, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
            self._updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self._global_step)
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep = 0)

    def setup_embeddings(self):
        with vs.variable_scope("Embedding"):
            if self._word2vec != None:
                self._enc_embed = tf.get_variable(name="enc_embed", shape=self._word2vec.shape, initializer = tf.constant_initializer(self._word2vec), trainable= False)
            else:
                self._enc_embed = tf.get_variable("enc_embed", [self._vocab_size, 512])

            if self._c_word2vec != None:
                self._c_enc_embed = tf.get_variable(name="c_enc_embed", shape=self._c_word2vec.shape, initializer = tf.constant_initializer(self._c_word2vec), trainable= False)
            else:
                self._c_enc_embed = tf.get_variable("c_enc_embed", [self._vocab_size, 512])


            if self._w_mask != None:
                self._wmask = tf.get_variable(name = "w_mask", shape = self._w_mask.shape, initializer = tf.constant_initializer(self._w_mask), trainable = False)
            else:
                self._wmask = tf.ones(shape = [1999,1999], trainable = False)

            self._encoder_inputs = embedding_ops.embedding_lookup(self._enc_embed, self._question_tokens)
            self._c_encoder_inputs = embedding_ops.embedding_lookup(self._c_enc_embed, self._c_question_tokens)

    def setup_ACM(self):
        with vs.variable_scope("C_ACM") as scope:
           inputs = self._encoder_inputs
           train = True

           self._ACM_RNNcell = FilterRNN(self._state_size, self._W, self._V)
           self._ACM_attCNN_cell = attenCNN(self._state_size, self._batch_size, trainable = train)

           initial_state = tf.Variable(tf.zeros([self._batch_size, self._state_size * 20]), trainable = False)

           outputs, output_state = rnn.dynamic_rnn(self._ACM_RNNcell, inputs, dtype = dtypes.float32, scope = "AC_RNN", initial_state = initial_state)
           outputs, output_state = rnn.dynamic_rnn(self._ACM_attCNN_cell, outputs, dtype = dtypes.float32, scope = "AC_CNN")

           max_out = tf.reduce_max(outputs, 1)
           mean_out = tf.reduce_mean(outputs, 1)
           self._ACM_weight = self._ACM_attCNN_cell.dump()
           self._ACM_final_state = max_out + mean_out
           self._ACM_output = outputs

    def setup_encoder(self):
        with vs.variable_scope("Encoder", initializer = tf.orthogonal_initializer()) as scope:
           inputs = self._encoder_inputs
           inputs_bw = tf.reverse_sequence(inputs, self._srclen, seq_dim=1, batch_dim=0)

           self._encoder_fw_cell = tf.contrib.rnn.GRUCell(self._state_size)
           self._encoder_bw_cell = tf.contrib.rnn.GRUCell(self._state_size)

           output_fw, output_state_fw = rnn.dynamic_rnn(self._encoder_fw_cell, inputs, dtype = dtypes.float32, scope = "FW")
           output_bw, output_state_bw = rnn.dynamic_rnn(self._encoder_bw_cell, inputs_bw, dtype = dtypes.float32, scope = "BW")

           outputs = tf.concat([output_fw, output_bw], axis = 2)
           output_states = tf.concat([output_state_fw, output_state_bw], axis = 1)
           self._encoder_output = outputs
           max_out = tf.reduce_max(outputs, 1)
           mean_out = tf.reduce_mean(outputs, 1)
           self._encoder_final_state = max_out + mean_out

    def setup_c_encoder(self):
        with vs.variable_scope("C_Encoder", initializer = tf.orthogonal_initializer()) as scope:
           inputs = self._c_encoder_inputs
           inputs_bw = tf.reverse_sequence(inputs, self._srclen, seq_dim=1, batch_dim=0)

           self._encoder_fw_cell = tf.contrib.rnn.GRUCell(self._state_size)
           self._encoder_bw_cell = tf.contrib.rnn.GRUCell(self._state_size)

           output_fw, output_state_fw = rnn.dynamic_rnn(self._encoder_fw_cell, inputs, dtype = dtypes.float32, scope = "FW")
           output_bw, output_state_bw = rnn.dynamic_rnn(self._encoder_bw_cell, inputs_bw, dtype = dtypes.float32, scope = "BW")

           outputs = tf.concat([output_fw, output_bw], axis = 2)
           output_states = tf.concat([output_state_fw, output_state_bw], axis = 1)

           self._c_encoder_output = outputs

           max_out = tf.reduce_max(outputs, 1)
           mean_out = tf.reduce_mean(outputs, 1)
           self._c_encoder_final_state = max_out + mean_out

    def setup_mark(self):
        input = self._mark
        train = False

        with vs.variable_scope("Mark_1") as scope:
             W1 = tf.get_variable('W1', [1999, 1999], trainable = train)
             b1 = tf.get_variable('b1', [1999], initializer = tf.uniform_unit_scaling_initializer(), trainable = train)
#             W1 = tf.get_variable(name="W1", shape=iter_utils.loadWData("mask_W1.npy").shape, initializer = tf.constant_initializer(iter_utils.loadWData("mask_W1.npy")), trainable= False)
#             b1 = tf.get_variable(name="b1", shape=iter_utils.loadWData("mask_b1.npy").shape, initializer = tf.constant_initializer(iter_utils.loadWData("mask_b1.npy")), trainable= False)
             result1 = tf.matmul(input, W1) + b1
             self._mask_W1 = W1
             self._mask_b1 = b1

        with vs.variable_scope("Mark_2",) as scope:
             W2 = tf.get_variable('W2', [1999, 512], trainable = train)
             b2 = tf.get_variable('b2', [512], initializer = tf.uniform_unit_scaling_initializer(), trainable = train)
#             W2 = tf.get_variable(name="W2", shape=iter_utils.loadWData("mask_W2.npy").shape, initializer = tf.constant_initializer(iter_utils.loadWData("mask_W2.npy")), trainable= False)
#             b2 = tf.get_variable(name="b2", shape=iter_utils.loadWData("mask_b2.npy").shape, initializer = tf.constant_initializer(iter_utils.loadWData("mask_b2.npy")), trainable= False)
             result2 = tf.matmul(result1, W2) + b2
             result2 = prelu(result2)
             self._mask_W2 = W2
             self._mask_b2 = b2

        self._mark_outputs = result2

    def setup_CNN(self):
        with vs.variable_scope("CNN") as scope:
            input = self._encoder_inputs
            input_4D = tf.expand_dims(input, -1)    #yields [batch x r x d x 1]

            pooled_outputs = []
            filter_sizes = [2,3,4,5]

            for i, filter_size in enumerate(filter_sizes):
                with vs.variable_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, 256, 1, self._state_size]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self._state_size]), name="b")
                    conv = tf.nn.conv2d(
                        input_4D,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")

                    h = prelu(tf.nn.bias_add(conv, b))
                    max_pooled = tf.reduce_max(h, axis = 1, keep_dims = True)
                    avg_pooled = tf.reduce_mean(h, axis = 1, keep_dims = True)
                    pooled = max_pooled + avg_pooled
                    pooled_outputs.append(pooled)

            num_filters_total = self._state_size * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, axis = 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            self._CNN_output = h_pool_flat

    def setup_c_CNN(self):
        with vs.variable_scope("c_CNN") as scope:
            input = self._c_encoder_inputs
            input_4D = tf.expand_dims(input, -1)    #yields [batch x r x d x 1]

            pooled_outputs = []
            filter_sizes = [3,4,5,6]

            for i, filter_size in enumerate(filter_sizes):
                with vs.variable_scope("conv-maxpool-%s" % filter_size) as scope:
                    filter_shape = [filter_size, 256, 1, self._state_size]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self._state_size]), name="b")
                    conv = tf.nn.conv2d(
                        input_4D,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")

                    h = prelu(tf.nn.bias_add(conv, b))

                    max_pooled = tf.reduce_max(h, axis = 1, keep_dims = True)
                    avg_pooled = tf.reduce_mean(h, axis = 1, keep_dims = True)
                    pooled = max_pooled + avg_pooled
                    pooled_outputs.append(pooled)

            num_filters_total = self._state_size * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, axis = 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            self._c_CNN_output = h_pool_flat

    def setup_decoder(self):
#        input = self._ACM_final_state
        input = self._CNN_output
#        mark = batch_norm_wrapper(self._mark_outputs, not self._forward_only, decay = 0.99)
#        input = tf.concat([input, self._c_encoder_final_state], axis = 1)
#        input = tf.concat([input, self._CNN_output], axis = 1)
        input = tf.concat([input, self._c_CNN_output], axis = 1)
        input = tf.concat([input, self._mark], axis = 1)
        input = batch_norm_wrapper(input, not self._forward_only, decay = 0.99)

        with vs.variable_scope("Decoder_1") as scope:
             W1 = tf.get_variable('W1', [6095, 512])
             b1 = tf.get_variable('b1', [512])
             result1 = tf.matmul(input, W1) + b1
             result1 = selu(result1)
             result1 = dropout_selu(result1, 0.05, training = not self._forward_only)

        with vs.variable_scope("Mask_6") as scope:
             W6 = tf.get_variable("W6", [512, 1999])
             b6 = tf.get_variable("b6", [1999])
             result = tf.matmul(result1, W6) + b6

        self._decoder_outputs = result

    def setup_decoder1(self):
        input = self._mark_outputs
        input = batch_norm_wrapper(input, not self._forward_only, decay = 0.99)

        with vs.variable_scope("Mask_2") as scope:
             W4 = tf.get_variable("W4", [512, 1999])
             b4 = tf.get_variable("b4", [1999])
             result = tf.matmul(input, W4) + b4

        self._decoder1_outputs = result

    def setup_decoder2(self):
        input = self._ACM_final_state
        input = batch_norm_wrapper(input, not self._forward_only, decay = 0.99)

        with vs.variable_scope("Mask_3") as scope:
             W4 = tf.get_variable("W4", [1024, 1999])
             b4 = tf.get_variable("b4", [1999])
             result = tf.matmul(input, W4) + b4

        self._decoder2_outputs = result

    def setup_loss(self):
        logits = self._decoder_outputs
        label = tf.to_float(self._labels)
        self._result = tf.nn.sigmoid(logits)

        weight = tf.get_variable(name = "weight", shape = self._weight.shape, initializer = tf.constant_initializer(self._weight), trainable = False)
        tiled_weight = tf.tile(weight, [self._batch_size, 1])

        loss1 = -tf.reduce_sum( ( (label*tf.log(self._result + 1e-9)) + ((1-label) * tf.log(1 - self._result + 1e-9)) )  , axis = 1, name='xentropy' )

        predicted = tf.round(tf.reshape(self._result, [-1]))
        label = tf.reshape(label, [-1])

        TP = tf.to_float(tf.count_nonzero(predicted * label)) + 1e-9
        TN = tf.to_float(tf.count_nonzero((predicted - 1) * (label - 1))) + 1e-9
        FP = tf.to_float(tf.count_nonzero(predicted * (label - 1))) + 1e-9
        FN = tf.to_float(tf.count_nonzero((predicted - 1) * label)) + 1e-9
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        self._loss = tf.reduce_mean(loss1)
        self._accurancy = f1

    def setup_summary(self):
        loss_summary = tf.summary.scalar("loss", self._loss)
        acc_summary = tf.summary.scalar("accuracy", self._accurancy)
        self._train_summary_op = tf.summary.merge([loss_summary, acc_summary])

    def train(self, session, question_tokens, labels, srclen, global_step, mark, c_data, c_seq):
        input_feed = {}
        input_feed[self._question_tokens] = question_tokens
        input_feed[self._c_question_tokens] = c_data
        input_feed[self._labels] = labels
        input_feed[self._mark] = mark
        input_feed[self._srclen] = srclen
        input_feed[self._c_srclen] = c_seq
        input_feed[self._keep_prob] = self._keep_prob_config
        input_feed[self._coeff] = self._coeff_config
        output_feed = [self._updates, self._loss, self._accurancy, self._result, self._train_summary_op]
        output = session.run(output_feed, input_feed)

        return output[1], output[2], output[3], output[4]

    def validate(self, session, question_tokens, labels, srclen, global_step, mark, c_data, c_seq):
        input_feed = {}
        input_feed[self._question_tokens] = question_tokens
        input_feed[self._c_question_tokens] = c_data
        input_feed[self._labels] = labels
        input_feed[self._mark] = mark
        input_feed[self._srclen] = srclen
        input_feed[self._c_srclen] = c_seq
        input_feed[self._keep_prob] = self._keep_prob_config
        input_feed[self._coeff] = self._coeff_config
        output_feed = [self._loss, self._accurancy, self._result]
        output = session.run(output_feed, input_feed)

        return output[0], output[1], output[2]

    def test(self, session, question_tokens, srclen, global_step, mark, c_data, c_seq):
        input_feed = {}
        input_feed[self._mark] = mark
        input_feed[self._question_tokens] = question_tokens
        input_feed[self._c_question_tokens] = c_data
        input_feed[self._srclen] = srclen
        input_feed[self._c_srclen] = c_seq
        input_feed[self._keep_prob] = 1.0
        input_feed[self._coeff] = self._coeff_config

        output_feed = [self._result]
        output = session.run(output_feed, input_feed)

        return output[0]

    def dumpMark(self, session):
        output_feed = [self._mask_W1, self._mask_b1, self._mask_W2, self._mask_b2]
        output = session.run(output_feed)
        return output

    def dumpAtt(self, session):
        att_attributes = self._ACM_attCNN_cell.dump()
        output_feed = [att_attributes]
        output = session.run(output_feed)
        return output[0]

    def setup_glu(self):
        def glu(kernel_shape, layer_input, layer_name, residual=None):
            """ Gated Linear Unit """
            # Pad the left side to prevent kernels from viewing future context
            kernel_width = kernel_shape[1]
            left_pad = kernel_width - 1
            paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
            padded_input = tf.pad(layer_input, paddings, "CONSTANT")

            # Kaiming intialization
            stddev = np.sqrt(2.0 / (kernel_shape[1] * kernel_shape[2]))

            # First conv layer
            W_g = tf.Variable(stddev, dtype=tf.float32)
            W_v = tf.Variable(tf.random_normal(kernel_shape, stddev=stddev), name="W%s" % layer_name)
            W =  (W_g / tf.nn.l2_normalize(W_v, 0)) * W_v
            b = tf.Variable(tf.zeros([kernel_shape[2] * kernel_shape[3]]), name="b%s" % layer_name)
            conv1 = tf.nn.depthwise_conv2d(
                padded_input,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv1")
            conv1 = tf.nn.bias_add(conv1, b)

            # Second gating sigmoid layer
            V_g = tf.Variable(stddev, dtype=tf.float32)
            V_v = tf.Variable(tf.random_normal(kernel_shape, stddev=stddev), name="V%s" % layer_name)
            V = (V_g / tf.nn.l2_normalize(V_v, 0)) * V_v
            c = tf.Variable(tf.zeros([kernel_shape[2] * kernel_shape[3]]), name="c%s" % layer_name)
            conv2 = tf.nn.depthwise_conv2d(
                padded_input,
                V,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv2")
            conv2 = tf.nn.bias_add(conv2, c)

            # Preactivation residual
            if residual is not None:
                conv1 = tf.add(conv1, residual)
                conv2 = tf.add(conv2, residual)

            h = tf.multiply(conv1, tf.sigmoid(conv2, name="sig"))

            return h

        input = tf.expand_dims(self._c_encoder_inputs, 1)
        kernel_shape = [1, 3, 256, 1]
        h0 = glu(kernel_shape, input, 0)
        h1 = glu(kernel_shape, h0, 1)
        h2 = glu(kernel_shape, h1, 2)
        h3 = glu(kernel_shape, h2, 3)
        h4 = glu(kernel_shape, h3, 4, h0)

        kernel_shape = [1, 3, 256, 1]
        h4a = glu(kernel_shape, h4, '14a')

        kernel_shape = [1, 3, 256, 1]
        h5 = glu(kernel_shape, h4a, 5)
        h6 = glu(kernel_shape, h5, 6)
        h7 = glu(kernel_shape, h6, 7)
        h8 = glu(kernel_shape, h7, 8)
        h9 = glu(kernel_shape, h8, 9, h4a)

        mean_pool = tf.reduce_mean(h9, axis = 2)
        max_pool = tf.reduce_max(h9, axis = 2)
        result = mean_pool + max_pool

        self._glu_output = tf.reshape(result, [self._batch_size, 256])



#dataTagger(410000, 256, 250, 10, 1.0, 0.01, 0.95, 1.0, 1.0)

""" 
        logit_list = tf.unstack(logits, axis = 0)
        label_list = tf.unstack(logits, axis = 0)

        total_F1 = []
        for logit_row, label_list in zip(logit_list, label_list):
            label_list = tf.to_int32(label_list)
            val, index = tf.nn.top_k(logit_row, 5)
            val_list = tf.unstack(val)

            p_list= []
            recall = None
            count = 1
            for j in val_list:
                pred = tf.greater_equal(logit_row, j)
                preds = tf.to_int32(tf.where(pred))
                TP = tf.to_float(tf.count_nonzero(preds * label_list)) + 1e-9
                TN = tf.to_float(tf.count_nonzero((preds - 1) * (label_list - 1))) + 1e-9
                FP = tf.to_float(tf.count_nonzero(preds * (label_list - 1))) + 1e-9
                FN = tf.to_float(tf.count_nonzero((preds - 1) * label_list)) + 1e-9

                precision = TP / (TP + FP)
                p_list.append(precision/ math.log(count))
                recall = TP / (TP + FN)
                count += 1

            prec = tf.reduce_sum(tf.stack(p_list))
            F1 = prec * recall / (prec + recall)
            total_F1.append(F1)

        accurancy = tf.reduce_mean(tf.stack(total_F1))
"""



""" 
            w_atten = tf.get_variable("w_atten", [self._num_units, 1])
            w_atten_x = tf.expand_dims(w_atten, 0)
            w_att = tf.tile(w_atten_x, [self._batch_size, 1, 1])
            att_para = tf.abs(tf.matmul(reshaped, w_att))                                 #batch_szie x step x 1

            sum_ = tf.reduce_sum(att_para, 1)
            sumx = tf.tile(tf.expand_dims(sum_, 1), [1, 20, 1])
            att_para_div = tf.div(att_para, sumx)
            att_para_final = tf.tile(att_para_div, [1, 1, self._num_units])
            att_out = tf.multiply(reshaped, att_para_final)

            weights = {
                # 1x16 conv, 1 input, 8 outputs
                'wc1': tf.get_variable("wc1", shape=[1, 16, 1, 8], initializer=tf.ones_initializer()),
                'wc2': tf.get_variable("wc2", shape=[1, 32, 8, 16 ], initializer=tf.contrib.layers.xavier_initializer()),
                'wc3': tf.get_variable("wc3", shape=[1, 64, 16, 16], initializer=tf.contrib.layers.xavier_initializer()),
                # fully connected layer ( if need)
                'wd1': tf.get_variable("wd1", shape=[2*16*128, 512], initializer=tf.contrib.layers.xavier_initializer()),
                # 1024 inputs, 10 outputs (class prediction)
                #'out': tf.get_variable("out", shape=[10, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            }

            biases = {
                'bc1': tf.get_variable("bc1", shape = [8], initializer = tf.zeros_initializer()),
                'bc2': tf.get_variable("bc2", shape = [16], initializer = tf.zeros_initializer()),
                'bc3': tf.get_variable("bc3", shape = [16], initializer = tf.zeros_initializer()),
                'bd1': tf.get_variable("bd1", shape = [100], initializer = tf.zeros_initializer()),
                #'out': tf.Variable(tf.zeros([num_classes]))
            }

            x_4d = tf.reshape(att_out, shape = [-1, 20, self._num_units, 1])
            conv1 = self.conv2d(x_4d, weights['wc1'], biases['bc1'])
            conv1 = self.maxpool2d(conv1, k=2)

            conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
            conv2 = self.maxpool2d(conv2, k=2)

            conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])
            conv3 = self.maxpool2d(conv3, k=2)

            fc1 = tf.reshape(conv3, [self._batch_size, -1])
"""
