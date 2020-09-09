# _*_ coding: utf-8 _*_
"""
Time:     2020/9/9 17:50
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     cora_preprogress.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
import pandas as pd
import random
random.seed(1)



def data_preprogress(data_path, labels_path):
    with open(data_path, 'r') as f:
        data = pd.read_csv(data_path, sep='\t', header=None)

        nodes = data[:][0].tolist()
        labels = data[:][1434].tolist()

        labels_set = list(set(labels))
        labels_set.sort()

        cate = {}
        for index, label in enumerate(labels_set):
            cate[label] = index

        node2cate = []
        for line in labels:
            node2cate.append(cate[line])

    with open(labels_path, 'w') as f:
        for index, node in enumerate(nodes):
            f.write('{}\t{}\n'.format(node, node2cate[index]))


if __name__ == '__main__':
    data_path = './origin/cora.content'
    labels_path = './progressed/cora_labels.txt'
    data_preprogress(data_path, labels_path)
