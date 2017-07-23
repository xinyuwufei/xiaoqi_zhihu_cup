from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import json
import csv

import numpy as np
from six.moves import xrange
import tensorflow as tf

import dataTagger
import iter_utils
import logging
from train import *

def dumpMark():
    with tf.Session() as sess:
        model, step = create_model(sess, True)
        var_list = model.dumpMark(sess)
        path = os.getcwd()
        path += "/Bin/"
        np.save(path + "mask_W1", var_list[0])
        np.save(path + "mask_b1", var_list[1])
        np.save(path + "mask_W2", var_list[2])
        np.save(path + "mask_b2", var_list[3])

def dumpAtt():
    with tf.Session() as sess:
        model, step = create_model(sess, True)
        var_list = model.dumpAtt(sess)
        path = os.getcwd()
        path += "/Bin/"
        np.save(path + "att_w_omega", var_list[0])
        np.save(path + "att_b_omega", var_list[1])
        np.save(path + "att_u_omega", var_list[2])

        for i, data in enumerate(var_list[3]):
            np.save(path + "att_w_%d"%i, data)

        for i, data in enumerate(var_list[4]):
            np.save(path + "mask_b_%d"%i, data)

dumpAtt()
#dumpMark()
