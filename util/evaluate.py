# _*_ coding: utf-8 _*_
"""
Time:     2020/9/9 17:50
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     evaluate.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
from util.svm_classify import Classifier

"""
train and evaluate node classify task by SVM
"""


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break

        vec = l.strip().split('\t')
        X.append(vec[0])
        Y.append(vec[1])
    fin.close()
    return X, Y


def evaluate_embeddings(embeddings, label_file):
    X, Y = read_node_label(label_file)
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
    clf = Classifier(embeddings=embeddings)
    clf.split_train_evaluate(X, Y, tr_frac)
