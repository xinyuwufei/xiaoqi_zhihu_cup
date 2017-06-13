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

logging.basicConfig(level=logging.INFO)

def loadDict():
    path = os.getcwd()
    file_name = path + '/Data/Eval/vocab.txt'
    store = []

    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            if line != "":
                store.append(line)
    return store

def writeCSV(data):
    path = os.getcwd()
    file_name = path + '/Obj/result.csv'
    writer = csv.writer(open(file_name, 'w'))
    for row in data:
        writer.writerow(row)

def pad(data):
    if len(data) < 6:
        diff = 6 - len(data)
        for i in range(diff):
            data.append(411720)
    return data

def buildMark(mark):
    length = len(mark)
    holder = np.zeros(shape = [length, 1999])
    count = 0
    for i in mark:
        for j in i:
            holder[count][j] = 1
        count += 1
    return holder

path = os.getcwd()
file_name = path + "/Data/Eval/name.txt"
data_name = path + "/Data/Eval/data.txt"
mark_name = path + "/Data/Eval/mark.txt"
dict = loadDict()
name_list = iter_utils.loadName(file_name)
data_list = iter_utils.loadData(data_name)
mark_list = iter_utils.loadData(mark_name)

holder = []
count = 0
count1 = 0

with tf.Session() as sess:
    model, step = create_model(sess, True)
    length = len(name_list)

    for i in range(length):
        temp = []
        data, name, mark = data_list[i], name_list[i], mark_list[i]
        mark = buildMark([mark])
        if(len(data) > 0):
            pad_d = pad(data)
            result = model.test(sess, np.asarray([pad_d]), [len(data)], 1, step, mark)
            result = max_n(result[0], 5)
            temp.append(str(name))
            for prob, num in reversed(result):
                temp.append(str(dict[num]))
            holder.append(temp)
        else:
            count1 += 1
            temp.append(str(name))
            temp.append(str(-1))
            temp.append(str(-1))
            temp.append(str(-1))
            temp.append(str(-1))
            temp.append(str(-1))
            holder.append(temp)
        if count % 10000 == 0:
            print("Procee " + str(count))
        count += 1

print(count)
print(count1)
writeCSV(holder)

""" 
    test_iterator = iter_utils.batch_iterator("test", FLAGS.batch_size)
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
"""
