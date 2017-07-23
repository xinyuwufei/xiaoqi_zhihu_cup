import os
import numpy as np
import math

def max_n(arr, n):
    indices = arr.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, arr.shape) for i in indices)
    return [[arr[i], i[0]] for i in indices]

def split_label():
    path = os.getcwd()
    file_name = path + "/Data/label"
    data = []
    count = 0
    file_count = 0
    with open(file_name + ".txt", 'r') as f:
        for line in f:
            data.append(line)

            if count != 0 and count % 750000 == 0:
                with open(file_name + str(file_count) + ".txt", 'w') as w:
                    for i in data:
                        w.write(i)
                data = []
                file_count += 1

            count += 1

    with open(file_name + str(file_count) + ".txt", 'w') as w:
        for i in data:
            w.write(i)

def order_embedding():
    path = os.getcwd()
    file_name = path + "/Data/word_embedding.txt"
    out_name = path + "/Data/out_embedding.txt"
    voca_name = path + "/Data/embedding_list.txt"
    w_dict = {}

    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            if line != "":
                temp_list = line.split(" ")
                name = temp_list.pop(0)
                name = name.replace("w", "")
                w_dict[int(name)] = temp_list


    keylist = w_dict.keys()
    keylist = sorted(keylist)

    with open(voca_name, 'w') as w1:
        for i in keylist:
            w1.write(str(i) + '\n')

    with open(out_name, 'w') as w:
        for i in keylist:
            w.write(" ".join(w_dict[i]) + '\n')

#order_embedding()

def parseData(num):
    path = os.getcwd()
    voca_name = path + "/Data/embedding_list.txt"
    file_name = path + "/Data/data" + str(num) + ".txt"
#    file_name = path + "/Data/eval0.txt"
    rev = {}
    count = 0
    with open(voca_name, 'r') as f:
        for line in f:
            line = line.replace("\n", '')
            rev[line] = str(count)
            count += 1

    holder = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", '')
            temp_list = line.split(" ")
            temp = []
            for i in temp_list:
                if i != '':
                    if i in rev:
                        temp.append(rev[i])
                    else:
                        if i == '|':
                            temp.append(i)
                        else:
                            temp.append(rev['-1'])

            holder.append(temp)

    with open(file_name, 'w') as w:
        for line in holder:
            w.write(" ".join(line) + '\n')

def countIdf(file_name, rev, count):
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", '')
            temp_list = line.split("|")
            list0 = temp_list[0].split(" ")
            list1 = temp_list[1].split(" ")
            final_list = list(set(list0 + list1))
            for i in final_list:
                if i != '':
                    if i in rev:
                        rev[int(i)] += 1
                    else:
                        rev[int(i)] = 1
            if(line != ''):
                count += 1
    return rev, count

def constructTfIdf():
    path = os.getcwd()
    voca_name = path + "/Data/c_embedding_list.txt"
    file0_name = path + "/Data/train/c_data.txt"
    file1_name = path + "/Data/test/c_data.txt"
    file2_name = path + "/Data/Eval/c_data.txt"

    rev = {}
    count = 0

    rev, count = countIdf(file0_name, rev, count)
    rev, count = countIdf(file1_name, rev, count)
    rev, count = countIdf(file2_name, rev, count)

    key_list = rev.keys()
    for key in sorted(key_list):
        rev[key] = math.log(rev[key] / count)

    return rev

def cleanData(rev, mode):
    path = os.getcwd()
    voca_name = path + "/Data/c_embedding_list.txt"
    file_name = path + "/Data/" + mode + "c_data.txt"
    out_name = path + "/Data/" + mode + "c_re_data.txt"
    holder = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", '')
            if (line != ""):
                temp_list = line.split("|")
                list0 = temp_list[0].split(" ")
                list1 = temp_list[1].split(" ")
                temp_list = list0 + list1
                count_list = []
                for i in temp_list:
                    if i != '':
                        count_list.append(int(i))
                final_list = list(set(count_list))
                final_list.sort(key = count_list.index)
                score_list = []
                length = len(count_list)
                for i in final_list:
                    ocr = count_list.count(i)
                    tf = ocr / length
                    score_list.append(rev[i] * tf)
                result = max_n(np.asarray(score_list), 20)
                result.sort(key = lambda x: x[1])
                temp = []
                for score, index in result:
                    temp.append(str(final_list[index]))
                holder.append(temp)

    with open(out_name, 'w') as w:
        for i in holder:
            w.write(' '.join(i) + '\n')


rev = constructTfIdf()
cleanData(rev, "train/")
cleanData(rev, "test/")
cleanData(rev, "Eval/")



