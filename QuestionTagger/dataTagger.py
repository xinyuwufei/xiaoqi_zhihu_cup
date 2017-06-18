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

class dataTagger(object):
    def __init__(self, batch_size, vocab_size, state_size, att_state_size, sen_state_size, max_gradient_norm, learning_rate,
                 learning_rate_decay_factor, coeff_config, dropout, weight, forward_only = False, optimizer = "adam",
                 word2vec = None):
        self._word2vec = word2vec
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
        self._label_len = 0
        self._weight = weight
        self._global_step = tf.Variable(0, trainable=False)
        self._keep_prob = tf.placeholder(tf.float32)
        self._coeff = tf.placeholder(tf.float32)
        self._labelLen = tf.placeholder(tf.int32)
        self._srclen = tf.placeholder(tf.int32, shape = [batch_size])
        self._question_tokens = tf.placeholder(tf.int32, shape = [batch_size, None])
        self._mark = tf.placeholder(tf.float32, shape = [batch_size, 1999])
        self._labels = tf.placeholder(tf.int32, shape = [batch_size, 1999])

        with tf.variable_scope("dataTagger", initializer = tf.uniform_unit_scaling_initializer()):
#        with tf.variable_scope("dataTagger", initializer = tf.orthogonal_initializer()):
            self.setup_embeddings()
            self.setup_encoder()
            self.setup_mark()
#            self.setup_attention()
            self.setup_CNN()
            self.setup_decoder()
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
            self._encoder_inputs = embedding_ops.embedding_lookup(self._enc_embed, self._question_tokens)

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
#           self._encoder_final_state = output_states

    def setup_attention(self):
        with vs.variable_scope("Attention") as scope:
           input = tf.transpose(self._encoder_output, perm = [0, 2, 1])
           W1 = tf.get_variable('W1', [self._att_state_size, 2 * self._state_size])
           W2 = tf.get_variable('W2', [self._sen_state_size, self._att_state_size])

           input = tf.reshape(input, [2 * self._state_size, -1])
           first_layer = tf.tanh(tf.matmul(W1, input))
           second_layer = tf.matmul(W2, first_layer)
           second_layer = tf.reshape(second_layer, [self._batch_size, self._sen_state_size, -1])
           self._attention =  tf.nn.softmax(second_layer)
           self._sentence = tf.reduce_max(tf.matmul(self._attention, self._encoder_output), 1)

    def setup_mark(self):
        input = self._mark

        with vs.variable_scope("Mark_1") as scope:
             W1 = tf.get_variable('W1', [1999, 1999])
             b1 = tf.get_variable('b1', [1999], initializer = tf.uniform_unit_scaling_initializer())
             result1 = tf.matmul(input, W1) + b1

        with vs.variable_scope("Mark_2",) as scope:
             W2 = tf.get_variable('W2', [1999, 512])
             b2 = tf.get_variable('b2', [512], initializer = tf.uniform_unit_scaling_initializer())
             result2 = tf.matmul(result1, W2) + b2
             result2 = prelu(result2)
#             result2 = selu(result2)
#             result2 = dropout_selu(result2, 0.1, training = not self._forward_only)

#        with vs.variable_scope("Mark_3") as scope:
#             W3 = tf.get_variable('W3', [512, 512])
#             b3 = tf.get_variable('b3', [512])
#             result = tf.matmul(result2, W3) + b3
#             result = prelu(result)

        self._mark_outputs = result2

    def setup_CNN(self):
        with vs.variable_scope("CNN") as scope:
            input = self._encoder_inputs
            input_4D = tf.expand_dims(input, -1)    #yields [batch x r x d x 1]

            pooled_outputs = []
            filter_sizes = [2,3,4,5]

            for i, filter_size in enumerate(filter_sizes):
                filter_shape = [filter_size, 256, 1, 256]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                conv = tf.nn.conv2d(
                    input_4D,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                with vs.variable_scope(str(i)) as scope:
                    h = prelu(tf.nn.bias_add(conv, b))

                max_pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',

                    name="pool")
                max_pooled = tf.reduce_max(max_pooled, axis = 1, keep_dims = True)


                avg_pooled = tf.nn.avg_pool(
                    h,
                    ksize=[1, 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',

                    name="pool")
                avg_pooled = tf.reduce_mean(avg_pooled, axis = 1, keep_dims = True)

                pooled = max_pooled + avg_pooled

                pooled_outputs.append(pooled)

            num_filters_total = 256 * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, axis = 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            self._CNN_output = h_pool_flat

    def setup_decoder(self):
        input = tf.reshape(self._encoder_final_state, shape = [self._batch_size, self._state_size*2])
        input = tf.concat([input, self._mark_outputs], axis = 1)
        input = tf.concat([input, self._CNN_output], axis = 1)

        input = batch_norm_wrapper(input, not self._forward_only, decay = 0.99)

        with vs.variable_scope("Decoder_1") as scope:
             W1 = tf.get_variable('W1', [1536 + 1024, 1024])
             b1 = tf.get_variable('b1', [1024])
             result1 = tf.matmul(input, W1) + b1
             result1 = selu(result1)
             result1 = dropout_selu(result1, 0.05, training = not self._forward_only)
#             result1 = prelu(result1)

#        with vs.variable_scope("Decoder_2") as scope:
#             W2 = tf.get_variable('W2', [1024, 1024])
#             b2 = tf.get_variable('b2', [1024])
#             result2 = tf.matmul(result1, W2) + b2
#             result2 = selu(result2)
#             result2 = dropout_selu(result2, 0.05, training = not self._forward_only)
#             result1 = prelu(result1)

        with vs.variable_scope("Decoder_2") as scope:
             W3 = tf.get_variable('W3', [1024, 1999])
             b3 = tf.get_variable('b3', [1999])
             result = tf.matmul(result1, W3) + b3

             self._W3 = W3
             self._b3 = b3

        self._decoder_outputs = result

    def setup_loss(self):
        logits = self._decoder_outputs
        self._result = tf.nn.sigmoid(logits)
        label = tf.to_float(self._labels)

#        weight = tf.get_variable(name = "weight", shape = self._weight.shape, initializer = tf.constant_initializer(self._weight), trainable = False)
#        tiled_weight = tf.tile(weight, [self._batch_size, 1])

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

#        self._result = tf.nn.sigmoid(logits)
#        label = tf.reshape(tf.to_float(self._labels), [-1])
#        loss1 = -tf.reduce_sum( (  (label*tf.log(self._result + 1e-9)) + ((1-label) * tf.log(1 - self._result + 1e-9)) )  , name='xentropy' )


    def setup_summary(self):
        loss_summary = tf.summary.scalar("loss", self._loss)
        acc_summary = tf.summary.scalar("accuracy", self._accurancy)
        self._train_summary_op = tf.summary.merge([loss_summary, acc_summary])

    def train(self, session, question_tokens, labels, srclen, label_len, global_step, mark):
        input_feed = {}
        input_feed[self._labelLen] = label_len
        input_feed[self._question_tokens] = question_tokens
        input_feed[self._labels] = labels
        input_feed[self._mark] = mark
        input_feed[self._srclen] = srclen
        input_feed[self._keep_prob] = self._keep_prob_config
        input_feed[self._coeff] = self._coeff_config
        output_feed = [self._updates, self._loss, self._accurancy, self._result, self._train_summary_op]
        output = session.run(output_feed, input_feed)

        return output[1], output[2], output[3], output[4]

    def validate(self, session, question_tokens, labels, srclen, label_len, global_step, mark):
        input_feed = {}
        input_feed[self._labelLen] = label_len
        input_feed[self._question_tokens] = question_tokens
        input_feed[self._labels] = labels
        input_feed[self._mark] = mark
        input_feed[self._srclen] = srclen
        input_feed[self._keep_prob] = self._keep_prob_config
        input_feed[self._coeff] = self._coeff_config
        output_feed = [self._loss, self._accurancy, self._result]
        output = session.run(output_feed, input_feed)

        return output[0], output[1], output[2]

    def test(self, session, question_tokens, srclen, label_len, global_step, mark):
        input_feed = {}
        input_feed[self._labelLen] = label_len
        input_feed[self._mark] = mark
        input_feed[self._question_tokens] = question_tokens
        input_feed[self._srclen] = srclen
        input_feed[self._keep_prob] = 1.0
        input_feed[self._coeff] = self._coeff_config

        output_feed = [self._result]
        output = session.run(output_feed, input_feed)

        return output[0]

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

