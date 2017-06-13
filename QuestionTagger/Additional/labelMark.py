import os

def tokenizer(string):
    return [int(i) for i in string.split()]

def readData(file_name):
    holder = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            if(len(line) > 0):
                line = line.split("|")
                result = line[0] + line[1]
                holder.append(tokenizer(result))
    return holder

def readLabel():
    path = os.getcwd()
    file_name = path + "/label_info.txt"
    holder = []

    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            if(len(line) != 0):
                line = line.replace(",", " ")
                holder.append(tokenizer(line))
    return holder

def compare(num):
    path = os.getcwd()
    file_name = path + "/eval" + str(num) + ".txt"

    data = readData(file_name)
    label = readLabel()

    holder = []
    count = 0

    for i in data:
        count += 1
        temp = []
        index = 0
        for j in label:
            flag = 0
            for k in j:
                if k not in i:
                    flag = 1
            if flag == 0:
                temp.append(str(index))
            index += 1
        if(len(temp) == 0):
            temp.append(str(-1))
        holder.append(temp)
        if count % 10000 == 0:
            print("processed at " + str(count))
    with open(path + "/eval_mark_" + str(num) + ".txt", 'w') as w:
        for i in holder:
            w.write(" ".join(i) + "\n")

compare(0)

