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
    normalized_probs = [float(pow(u_prob, 0.75)) / norm_const for u_prob in unnormalized_probs]

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


def preprocess_transition_probs(G, is_directed, ):
    '''
    Preprocessing of transition probabilities for guiding the random walks.
    '''
    alias_nodes = {}
    for node in G.nodes():
        unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        alias_nodes[node] = alias_setup(normalized_probs)

    alias_edges = {}

    if is_directed:
        for edge in G.edges():
            alias_edges[edge] = get_alias_node(edge[0])
    else:
        for edge in G.edges():
            alias_edges[edge] = get_alias_node(edge[0])
            alias_edges[(edge[1], edge[0])] = get_alias_node(edge[1])

    return alias_nodes, alias_edges


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

    def __init__(self, G, J, q, nodes, node2index, index2edge, num_negativate):
        print('Graph Creating...')
        self.G = G
        self.J = J
        self.q = q
        self.nodes = nodes
        self.node2index = node2index
        self.index2edge = index2edge
        self.num_negative = num_negativate

    def __len__(self):
        return len(self.G.edges())

    def __getitem__(self, index):
        edge_index_in_alias = alias_draw(J=self.J, q=self.q)
        edge = self.index2edge[edge_index_in_alias]

        # postive
        node_i, node_j = edge[0], edge[1]

        # negativate
        neighbor_node_i = list(self.G.neighbors(node_i))

        negative_sample = []
        K = self.num_negative
        while K:
            negative = random.choice(self.nodes)
            if negative not in neighbor_node_i:
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
    def __init__(self, args, G, J, q, nodes, node2index, index2edge):
        self.args = args
        self.G = G
        self.J = J
        self.q = q
        self.nodes = nodes
        self.node2index = node2index
        self.index2edge = index2edge

    def TrainLoader(self):
        train_loader = data.DataLoader(
            GraphTrainDataSet(self.G, self.J, self.q, self.nodes, self.node2index, self.index2edge,
                              self.args.num_negative),
            batch_size=self.args.batch_size, shuffle=True, num_workers=0,
            pin_memory=False, drop_last=False)
        return train_loader
