# _*_ coding: utf-8 _*_
"""
Time:     2020/9/9 19:43
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     dataloader.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
import networkx as nx


def read_graph(args):
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G
