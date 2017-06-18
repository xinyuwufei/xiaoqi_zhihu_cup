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


def pad(data, seq):
    max_len = max(max(seq), 6)
    result = np.zeros(shape = [FLAGS.batch_size, max_len])
    result.fill(411720)
    length = len(result)
    for i in range(length):
        result[i][:seq[i]] = data[i]
    return result

def buildMark(mark):
    length = len(mark)
    holder = np.zeros(shape = [length, 1999])
    count = 0
    for i in mark:
        for j in i:
            holder[count][j] = 1
        count += 1
    return holder

def getSeq(data):
    return [len(i) for i in data]


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
    validate(sess, model, step)

"""
    length = len(name_list)
    total = int(length / FLAGS.batch_size)

    for i in range(total):
        bot = count * FLAGS.batch_size
        roof = bot + FLAGS.batch_size
        data, name, mark = data_list[bot:roof], name_list[bot:roof], mark_list[bot:roof]
        mark = buildMark(mark)
        seq = getSeq(data)
        pad_d = pad(data, seq)
        t_result = model.test(sess, np.asarray(pad_d), (np.asarray(seq)).reshape(-1), FLAGS.batch_size, step, mark)

        check = 0
        for row in t_result:
            temp = []
            result = max_n(row, 5)
            temp.append(str(name[check]))
            for prob, num in reversed(result):
                temp.append(str(dict[num]))
            holder.append(temp)
            check += 1

        if count % 100 == 0:
            print("Procee " + str(count))
        count += 1


    bot = count * FLAGS.batch_size
    data, name, mark = data_list[bot:], name_list[bot:], mark_list[bot:]
    cut = len(data)
    diff = FLAGS.batch_size - len(data)
    data1, name1, mark1 = data_list[:diff], name_list[:diff], mark_list[:diff]
    data += data1
    name += name1
    mark += mark1

    mark = buildMark(mark)
    seq = getSeq(data)
    pad_d = pad(data, seq)

    t_result = model.test(sess, np.asarray(pad_d), (np.asarray(seq)).reshape(-1), FLAGS.batch_size, step, mark)
    t_result = t_result[:cut]
    check = 0
    for row in t_result:
        temp = []
        result = max_n(row, 5)
        temp.append(str(name[check]))
        for prob, num in reversed(result):
            temp.append(str(dict[num]))
        holder.append(temp)
        check += 1

print(len(holder))

writeCSV(holder)
""" 
