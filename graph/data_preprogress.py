# _*_ coding: utf-8 _*_
"""
Time:     2020/9/9 17:50
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     data_preprogress.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
def data_preprogress(data_path, save_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        edges = []
        for line in lines:
            edge = []
            start = 0
            if len(line.strip().split()) > 1:
                for i, v in enumerate(line.strip().split()):
                    if i == 0:
                        start = v
                        continue
                    edge.append((start, v))
            else:
                continue
            edges.extend(edge)
    with open(save_path, 'w') as f:
        for line in edges:
            f.write('{}\t{}\n'.format(line[0], line[1]))




if __name__ == '__main__':
    data_path = 'dblp/origin/dblp_adjedges.txt'
    save_path = 'dblp/progressed/dblp_adjedges.adjlist'
    data_preprogress(data_path, save_path)

