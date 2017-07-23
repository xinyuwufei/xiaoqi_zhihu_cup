#!/usr/bin/env python3
#-*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import xrange
import tensorflow as tf
import random
import os
import copy
import pandas

def tokenizer(string):
    return [int(i) for i in string.split()]

def loadLabel(file_name):
    data = []
    with open(file_name, "r") as f:
        for lines in f:
            temp = tokenizer(lines)
            data.append(temp)
    return data

def loadData(file_name):
    data = []
    with open(file_name, "r") as f:
        for lines in f:
            line = lines.split("|")[0] + lines.split("|")[1]
            temp = tokenizer(line)
            data.append(temp)
    return data

def loadDespData(file_name):
    data = []
    with open(file_name, "r") as f:
        for lines in f:
            lines = lines.split("|")[1]
            temp = tokenizer(lines)
            data.append(temp)
    return data

def loadName(file_name):
    data = []
    with open(file_name, 'r') as f:
        for lines in f:
            lines = lines.replace("\n", "")
            data.append(lines)
    return data

def loadWeight():
    path = os.getcwd()
    file_path = path + "/Data/weight.txt"
    holder = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            if line != "":
                holder.append(float(line))
    return np.asarray([holder])

class batch_iterator(object):
    def __init__(self, mode, batch_size):
        path = os.getcwd()
        data_path = path + "/Data/" + mode + "/data.txt"
        c_data_path = path + "/Data/" + mode + "/c_data.txt"
        label_path = path + "/Data/" + mode + "/label.txt"
        mark_path = path + "/Data/" + mode + "/mark.txt"
        self._epochs = 0
        self._data = loadData(data_path)
        self._c_data = loadData(c_data_path)
        self._mark = loadData(mark_path)
        self._label = loadLabel(label_path)
        self._batch_size = batch_size
        self._size = len(self._data)
        self._total = int(self._size / self._batch_size)
        self.reset()

    def reset(self):
        self._list = [i for i in range(0, self._total)]
        random.shuffle(self._list)

    def buildLabel(self, label):
        holder = np.zeros(shape = [self._batch_size, 1999], dtype = np.float32)
        length = len(label)
        for i in range(length):
            for j in label[i]:
                holder[i][j] = 1
        return holder

    def getSeq(self, data):
        return [len(i) for i in data]

    def pad(self, data, seq):
        max_len = max(max(seq), 6)
        result = np.zeros(shape = [self._batch_size, max_len])
        result.fill(411720)
        length = len(result)
        for i in range(length):
            result[i][:seq[i]] = data[i]
        return result

    def pad_c(self, data, seq):
        max_len = max(max(seq), 6)
        result = np.zeros(shape = [self._batch_size, max_len])
        result.fill(11973)
        length = len(result)
        for i in range(length):
            result[i][:seq[i]] = data[i]
        return result


    def buildMark(self, mark):
        length = len(mark)
        holder = np.zeros(shape = [length, 1999])
        count = 0
        for i in mark:
            for j in i:
                if j != -1:
                    holder[count][j] = 1
            count += 1
        return holder

    def next_batch(self):
        if not self._list:
            self._epochs += 1
            self.reset()
        current = self._list.pop() * self._batch_size
        roof = current + self._batch_size

        data = self._data[current:roof]
        seq = self.getSeq(data)
        data = self.pad(data, seq)
        seq = (np.asarray(seq)).reshape(-1)

        c_data = self._c_data[current:roof]
        c_seq = self.getSeq(c_data)
        c_data = self.pad_c(c_data, c_seq)
        c_seq = (np.asarray(c_seq)).reshape(-1)

        label = self._label[current:roof]
        label_ = self.buildLabel(label)

        mark = self._mark[current:roof]
        mark = self.buildMark(mark)

        return data, label_, label, seq, mark, c_data, c_seq

def loadWord2Vec(name):
    abs = os.getcwd()
    w2vec_part = abs + "/Data/" + name
    w2vec = np.load(w2vec_part)

#    data = pandas.read_csv(w2vec_part, header = None, sep = " ")
#    w2vec = data.as_matrix()

    return w2vec

loadWord2Vec("c_out_embedding.txt")

def loadMask():
    abs = os.getcwd()
    path = abs + "/Bin/mask.txt"
    array = np.genfromtxt(path, delimiter = " ")
    return array

def loadW():
    abs = os.getcwd()
    path = abs + "/Bin/W.txt"
    array = np.genfromtxt(path, delimiter = " ")
    return array

def loadV():
    abs = os.getcwd()
    path = abs + "/Bin/V.txt"
    array = np.genfromtxt(path, delimiter = " ")
    return array

def loadWData(name):
    abs = os.getcwd()
    path = abs + "/Bin/" + name
    array = np.load(path)
    return array


#loadWord2Vec()
#a =  batch_iterator("train", 2)
#b,c,d,e, o = a.next_batch()
#print(f)
