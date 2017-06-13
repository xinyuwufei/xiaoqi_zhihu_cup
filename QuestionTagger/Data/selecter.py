import os
import random
import math

def tokenizer(string):
    return [int(i) for i in string.split()]

def readData(name):
    path = os.getcwd()
    file_name = path + "/" + name
    holder = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            if len(line) != 0:
                holder.append(line)
    return holder

def writeData(name, mode, data):
    path = os.getcwd()
    file_name = path + "/" + mode + "/" + name
    with open(file_name, 'w') as w:
        for i in data:
            w.write(i + "\n")

def readLabel(name):
    path = os.getcwd()
    file_name = path + "/" + name
    holder = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            if len(line) != 0:
                holder.append(tokenizer(line))
    return holder

def writeLabel(name, mode, data):
    path = os.getcwd()
    file_name = path + "/" + mode + "/" + name
    with open(file_name, 'w') as w:
        for i in data:
            w.write(" ".join(str(x) for x in i) + "\n")

label = readLabel("label.txt")
imbal = readLabel("imbal.txt")
imbal = [i[0] for i in imbal]

count = 0
keep_list = []
sample_list = []

print("----------------------counting-----------------------------")
for i in label:
    intersec = [element for element in i if element in imbal]
    if(len(intersec) == 0):
        keep_list.append(count)
    else:
        sample_list.append(count)
    count += 1

    if count % 10000 == 0:
        print("processing at:" + str(count))

print("----------------------sampling-----------------------------")
random.shuffle(sample_list)
length = len(sample_list)
length = round(length * 0.75)
train_list= sample_list[0:length]
test_list = sample_list[length:]

total_list = train_list + keep_list + keep_list
random.shuffle(total_list)

print("----------------------rendering-----------------------------")

data0 = readData("data0.txt")
data1 = readData("data1.txt")
data2 = readData("data2.txt")
data3 = readData("data3.txt")
data = data0 + data1 + data2 + data3


mark0 = readData("label_mark_0.txt")
mark1 = readData("label_mark_1.txt")
mark2 = readData("label_mark_2.txt")
mark3 = readData("label_mark_3.txt")
mark = mark0 + mark1 + mark2 + mark3

train_label_list = []
train_data_list = []
train_mark_list = []
test_label_list = []
test_data_list = []
test_mark_list = []

for i in total_list:
    train_label_list.append(label[i])
    train_data_list.append(data[i])
    train_mark_list.append(mark[i])

writeData("data.txt", "train", train_data_list)
writeData("mark.txt", "train", train_mark_list)
writeLabel("label.txt", "train", train_label_list)

for i in test_list:
    test_label_list.append(label[i])
    test_data_list.append(data[i])
    test_mark_list.append(mark[i])

writeData("data.txt", "test", test_data_list)
writeData("mark.txt", "test", test_mark_list)
writeLabel("label.txt", "test", test_label_list)

#print(len(imbal))
