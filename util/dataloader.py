# _*_ coding: utf-8 _*_
"""
Time:     2020/9/9 19:43
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     dataloader.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
import random
import numpy as np
import networkx as nx
from torch.utils import data


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


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q  # index; prob


def get_alias_node(G):
    '''
    Get the alias node setup lists for a given node.
    '''
    node_degree = {}
    index2node = {}
    unnormalized_probs = []
    for index, node in enumerate(sorted(G.nodes())):
        index2node[index] = node
        node_degree[node] = len((list(G.neighbors(node))))
        unnormalized_probs.append(pow(node_degree[node], 0.75))

    norm_const = sum(unnormalized_probs)
    normalized_probs = [u_prob / norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs), index2node


def get_alias_edge(G):
    '''
    Get the alias edge setup lists for a given edge.
    '''
    index2edge = {}
    unnormalized_probs = []
    for index, edge in enumerate(sorted(G.edges())):
        index2edge[index] = (edge[0], edge[1])
        unnormalized_probs.append(G[edge[0]][edge[1]]['weight'])

    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs), index2edge


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


class GraphTrainDataSet(data.Dataset):
    """
        GraphTrainDataSet is employed to load train set
        Args:
    """

    def __init__(self, G, J_edge, q_edge, J_node, q_node, nodes, node2index, index2edge, index2node, num_negativate):
        self.G = G
        self.J_edge = J_edge
        self.q_edge = q_edge
        self.J_node = J_node
        self.q_node = q_node
        self.nodes = nodes
        self.node2index = node2index
        self.index2edge = index2edge
        self.index2node = index2node
        self.num_negative = num_negativate

    def __len__(self):
        return len(self.G.edges())

    def __getitem__(self, index):
        # postive
        edge_index_in_alias = alias_draw(J=self.J_edge, q=self.q_edge)
        edge = self.index2edge[edge_index_in_alias]
        node_i, node_j = edge[0], edge[1]
        # negativate
        neighbor_node_i = list(self.G.neighbors(node_i))
        negative_sample = []
        K = self.num_negative
        while K:
            # negative = random.choice(self.nodes)
            negative_index_in_alias = alias_draw(self.J_node, self.q_node)
            negative = self.index2node[negative_index_in_alias]
            if negative not in neighbor_node_i and negative not in negative_sample:
                negative_sample.append(negative)
                K -= 1
            elif negative in neighbor_node_i:
                continue
        # reduce dict size by change node number to index
        node_i = self.node2index[node_i]
        node_j = self.node2index[node_j]

        i_list = []
        j_list = []
        i_list.append(node_i)
        j_list.append(node_j)
        for neg in negative_sample:
            neg = self.node2index[neg]
            i_list.append(node_i)
            j_list.append(neg)
        return i_list, j_list


class NodeDataLoader():
    def __init__(self, args, G, J_edge, q_edge, J_node, q_node, nodes, node2index, index2node, index2edge):
        self.args = args
        self.G = G
        self.J_edge = J_edge
        self.q_edge = q_edge
        self.J_node = J_node
        self.q_node = q_node
        self.nodes = nodes
        self.node2index = node2index
        self.index2edge = index2edge
        self.index2node = index2node

    def TrainLoader(self):
        train_loader = data.DataLoader(
            GraphTrainDataSet(self.G, self.J_edge, self.q_edge, self.J_node, self.q_node, self.nodes, self.node2index,
                              self.index2edge, self.index2node, self.args.num_negative),
            batch_size=self.args.batch_size, shuffle=True, num_workers=0,
            pin_memory=False, drop_last=False)
        return train_loader
