#!/usr/bin/env python3
#-*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import json

import numpy as np
from six.moves import xrange
import tensorflow as tf

import dataTagger
import iter_utils

import logging
logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning_rate")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decay")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm")
tf.app.flags.DEFINE_float("coeff", 1.0, "loss coefficient")
tf.app.flags.DEFINE_float("dropout", 0.05, "dropout rate")
tf.app.flags.DEFINE_integer("epochs", 40, "number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 1, "size of batch")
tf.app.flags.DEFINE_integer("state_size", 512, "size of each model layer")
tf.app.flags.DEFINE_integer("att_state_size", 386, "size of each model layer")
tf.app.flags.DEFINE_integer("sen_state_size", 15, "size of each model layer")
tf.app.flags.DEFINE_integer("vocab_size",  411721, "size of voacab")
tf.app.flags.DEFINE_integer("max_len", 20, "maximum length")
tf.app.flags.DEFINE_string("log_dir", "/Log/", "log dir")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam/sgd")
tf.app.flags.DEFINE_integer("print_every", 1000, "How many iteration to print")

FLAGS = tf.app.flags.FLAGS

def create_model(session, forward_only):
    model = dataTagger.dataTagger(FLAGS.batch_size, FLAGS.vocab_size, FLAGS.state_size, FLAGS. att_state_size, FLAGS.sen_state_size, FLAGS.max_gradient_norm,
                                FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.coeff,
                                FLAGS.dropout, forward_only = forward_only, optimizer = FLAGS.optimizer, word2vec = iter_utils.loadWord2Vec())
    ckpt = tf.train.get_checkpoint_state(os.getcwd() + FLAGS.log_dir)
    if ckpt:
        logging.info("Reading model parameters from %s ." % ckpt.model_checkpoint_path)
        model._saver.restore(session, ckpt.model_checkpoint_path)
        step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
    else:
        logging.info("Create model with fresh parameter.")
        session.run(tf.global_variables_initializer())
        logging.info("Num params: %d" %sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        step = 0
    return model, step

def max_n(arr, n):
    indices = arr.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, arr.shape) for i in indices)
    return [[arr[i], i[0]] for i in indices]

def train():
    iterator = iter_utils.batch_iterator("train", FLAGS.batch_size)
    test_iterator = iter_utils.batch_iterator("test", FLAGS.batch_size)
    with tf.Session() as sess:
        best_epochs = 0
        previous_losses = 0
        exp_cost = None
        exp_length = None
        exp_norm = None
        total_iters = 0
        start_time = time.time()
        model, step = create_model(sess, False)
        current_step = step
        global_minimum = 10000000000
        train_accurancy = []
        checkpoint_path = os.getcwd() + os.path.join(FLAGS.log_dir, "checkpoint")

        train_summary_dir = os.path.join(os.getcwd() + "/Log/", "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        while(FLAGS.epochs == 0 or iterator._epochs < FLAGS.epochs):
            data, label, label_, seq, mark = iterator.next_batch()
            if(len(data[0]) == 0):
                data, label, label_, seq, mark = iterator.next_batch()
            tic = time.time()
            cost, accurancy, result, summary = model.train(sess, data, label, seq, len(label), step, mark)
            train_summary_writer.add_summary(summary, step)
            toc = time.time()
            iter_time = toc - tic
            train_accurancy.append(accurancy)

            total_iters += 2000
            tps = total_iters/ (time.time() - start_time)
            current_step += 1
            model._global_step = current_step
            lengths = len(label)
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)

            if not exp_cost:
                exp_cost = cost
                exp_length = mean_length
            else:
                exp_cost = 0.99 * exp_cost + 0.01*cost
                exp_length = 0.99 * exp_length + 0.01*mean_length

            cost = cost/mean_length

            if global_minimum != 0 and cost < global_minimum:
                best_epoch = current_step
                global_minimum = cost

            if current_step % FLAGS.print_every == 0:
                result = max_n(result[0], 5)
                print(result)
                print(label_[0])
                logging.info('epoch %d, iter %d, cost %f, exp_cost %f, tps %f, length mean/std %f/%f, accuancy %f' %
                            (iterator._epochs, current_step, cost, exp_cost / exp_length, tps, mean_length, std_length, sum(train_accurancy) / float(len(train_accurancy))))
                train_accurancy = []
                if current_step % 10000 == 0:
                    test_cost = 0
                    test_accurancy = 0
                    for k in range(1000):
                        data, label, label_, seq, mark = test_iterator.next_batch()
                        if(len(data[0]) == 0):
                            data, label, label_, seq, mark = test_iterator.next_batch()
                        cost, accurancy = model.validate(sess, data, label, seq, len(label), step, mark)
                        test_cost += cost
                        test_accurancy += accurancy
                    print(test_cost / 150000.0)
                    print(test_accurancy / 1000.0)

                    s = input("-->:")
                    if s == "1":
                        model._saver.save(sess, checkpoint_path, global_step= current_step)
            step += 1
            sys.stdout.flush()

def main():
    train()

if __name__ == "__main__":
    main()
