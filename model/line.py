# _*_ coding: utf-8 _*_
"""
Time:     2020/9/10 10:47
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     line.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from util.dataloader import read_graph, NodeDataLoader, get_alias_edge, get_alias_node

class Line(nn.Module):
    def __init__(self, dict_size, embed_dim=128, order="first"):
        super(Line, self).__init__()

        assert order in ["first", "second", "all"], print("Order should either be [first, second, all]")
        self.dict_size = dict_size
        self.embed_dim = embed_dim
        self.order = order
        self.embeddings = nn.Embedding(dict_size, embed_dim)
        self.context_embeddings = nn.Embedding(dict_size, embed_dim)
        self.embeddings.weight.data.uniform_(-0.5, 0.5)
        self.context_embeddings.weight.data.uniform_(-0.5, 0.5)

    def forward(self, nodeindex, v_i, v_j):
        # init embeddings
        if self.order == 'first':
            u_i = self.embeddings(torch.LongTensor(v_i))
            u_j = self.embeddings(torch.LongTensor(v_j))
            return u_i, u_j
        elif self.order == 'second':
            u_i = self.embeddings(torch.LongTensor(v_i))
            u_j_context = self.context_embeddings(torch.LongTensor(v_j))
            return u_i, u_j_context
        elif self.order == 'all':
            u_i = self.embeddings(torch.LongTensor(v_i))
            u_j1 = self.embeddings(torch.LongTensor(v_j))
            u_j2 = self.context_embeddings(torch.LongTensor(v_j))
            return u_i, u_j1, u_j2



def main(args):
    # load data
    G = read_graph(args)
    nodes = sorted(list(G.nodes()))
    J_q_edge, index2edge = get_alias_edge(G)  # alias edge table:(J,q)
    J_edge, q_edge = J_q_edge[0], J_q_edge[1]
    J_q_node, index2node = get_alias_node(G)  # alias node table:(J,q)
    J_node, q_node = J_q_node[0], J_q_node[1]
    node2index = dict(zip(nodes, range(len(nodes))))
    dict_size = len(nodes)
    node_index = torch.LongTensor(range(0, dict_size))

    NodeDataLoaderclass = NodeDataLoader(args=args, G=G, J_edge=J_edge, q_edge=q_edge, J_node=J_node, q_node=q_node, nodes=nodes,
                                         node2index=node2index, index2node=index2node, index2edge=index2edge)
    train_loader = NodeDataLoaderclass.TrainLoader()
    # model
    model = Line(dict_size, embed_dim=args.dimensions, order=args.order)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    for epoch in range(args.iter):
        total_batches = len(train_loader)

        pbar = tqdm(iterable=enumerate(train_loader), total=total_batches,
                    desc='Epoch {}/{}'.format(epoch, args.iter))
        loss_record = []
        for iteration, batch in pbar:
            v_i = batch[0]  # v_i=[pos_i,neg_i1,neg_i2,neg_i3,neg_i4,neg_i5]
            v_j = batch[1]  # v_i=[pos_j,neg_j1,neg_j2,neg_j3,neg_j4,neg_j5]

            loss = 0
            for i in range(len(v_i)):
                if args.order == 'all':
                    u_i, u_j1, u_j2 = model(node_index, v_i[i], v_j[i])
                    temp1 = torch.sum(torch.mul(u_i, u_j1), dim=1)
                    temp2 = torch.sum(torch.mul(u_i, u_j2), dim=1)
                    if i == 0:  # postive
                        loss1 = -torch.mean(F.logsigmoid(temp1), dim=0)
                        loss2 = -torch.mean(F.logsigmoid(temp2), dim=0)
                    else:  # negative
                        loss1 = -torch.mean(F.logsigmoid(-temp1), dim=0)
                        loss2 = -torch.mean(F.logsigmoid(-temp2), dim=0)
                    loss += (loss1 + loss2)
                else:
                    u_i, u_j = model(node_index, v_i[i], v_j[i])
                    temp = torch.sum(torch.mul(u_i, u_j), dim=1)
                    if i == 0:  # postive
                        temp = temp
                    else:  # negative
                        temp = -temp
                    loss += -torch.mean(F.logsigmoid(temp), dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(train_loss=float(loss))
            loss_record.append(loss)

    print('train loss is {}'.format(sum(loss_record) / len(loss_record)))

    if args.order == 'first':
        embeddings = model.embeddings.weight.data.numpy()
    elif args.order == 'second':
        embeddings = torch.add(model.embeddings.weight.data, model.context_embeddings.weight.data).numpy()
    elif args.order == 'all':
        first_emb = model.embeddings.weight.data.numpy()
        second_emb = torch.add(model.embeddings.weight.data, model.context_embeddings.weight.data).numpy()
        embeddings = np.concatenate((first_emb, second_emb), axis=1)

    if not os.path.exists(args.output_emb):
        os.makedirs(args.output_emb)
    emb_path = args.output_emb + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.emb'

    index2node = dict(zip(range(len(nodes)), nodes))

    np.savetxt(emb_path, embeddings, fmt='%1.8f')

    with open(emb_path, 'r') as f1:
        lines = f1.readlines()
    with open(emb_path, 'w+') as f2:
        for i, l in enumerate(lines):
            f2.write('{} {}'.format(index2node[i], l))

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    checkpoints_path = args.checkpoints_path + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.pth'
    print("Done training, saving model to {}".format(checkpoints_path))
    torch.save(model, "{}".format(checkpoints_path))
