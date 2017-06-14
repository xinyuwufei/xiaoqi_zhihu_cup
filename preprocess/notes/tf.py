# coding:utf-8
import os
import sys
import pandas as pd

train_data = pd.read_csv('../../data/question_train_set.txt', sep="\t", header = None,usecols=[2,4],na_filter=False)
train_data.columns = ["title_word_id", "description_word_id"]
eval_data = pd.read_csv('../../data/question_eval_set.txt', sep="\t", header = None,usecols=[2,4],na_filter=False)
eval_data.columns = ["title_word_id", "description_word_id"]
l=[]
i=0
for x in [train_data,eval_data]:
    for xx in x['title_word_id']:
        i+=1
        print(i)
        if xx.strip():
#             print(xx)
            l+=xx.strip().split(',')
    for xx in x["description_word_id"]:
        i+=1
        print(i)
        if xx.strip():
#             print(xx)
            l+=xx.strip().split(',')
cnt_srs = pd.Series(l).value_counts()
cnt_srs.to_csv(path='tf.csv')