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

from util.dataloader import read_graph, NodeDataLoader, get_alias_edge

class Line(nn.Module):
    def __init__(self, dict_size, embed_dim=128, order="first", num_negative=5):
        super(Line, self).__init__()

        assert order in ["first", "second", "all"], print("Order should either be [first, second, all]")
        self.dict_size = dict_size
        self.embed_dim = embed_dim
        self.order = order
        self.embeddings = nn.Embedding(dict_size, embed_dim)
        self.second_embeddings = nn.Embedding(dict_size, embed_dim)
        self.context_embeddings = nn.Embedding(dict_size, embed_dim)
        self.embeddings.weight.data.uniform_(-0.5, 0.5)
        self.second_embeddings.weight.data.uniform_(-0.5, 0.5)
        self.context_embeddings.weight.data.uniform_(-0.5, 0.5)

    def forward(self, nodeindex, v_i, v_j, device):
        # init embeddings
        if self.order == 'first':
            u_i = self.embeddings(torch.LongTensor(v_i)).to(device)
            u_j = self.embeddings(torch.LongTensor(v_j)).to(device)
            return u_i, u_j
        elif self.order == 'second':
            u_i = self.embeddings(torch.LongTensor(v_i)).to(device)
            u_j_context = self.context_embeddings(torch.LongTensor(v_j)).to(device)
            return u_i, u_j_context
        elif self.order == 'all':
            u_i1 = self.embeddings(torch.LongTensor(v_i)).to(device)
            u_j1 = self.embeddings(torch.LongTensor(v_j)).to(device)
            u_i2 = self.second_embeddings(torch.LongTensor(v_i)).to(device)
            u_j2 = self.context_embeddings(torch.LongTensor(v_j)).to(device)
            return u_i1, u_j1, u_i2, u_j2



def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load data
    G = read_graph(args)
    nodes = sorted(list(G.nodes()))
    J_q, index2edge = get_alias_edge(G)  # alias table:(J,q)
    J, q = J_q[0], J_q[1]
    node2index = dict(zip(nodes, range(len(nodes))))
    dict_size = len(nodes)
    node_index = torch.LongTensor(range(0, dict_size))

    NodeDataLoaderclass = NodeDataLoader(args=args, G=G, J=J, q=q, nodes=nodes,
                                         node2index=node2index, index2edge=index2edge)
    train_loader = NodeDataLoaderclass.TrainLoader()
    # model
    model = Line(dict_size, embed_dim=args.dimensions, order=args.order, num_negative=args.num_negative)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, lambd=1e-4, alpha=0.75, t0=1e6)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    for epoch in range(args.iter):
        total_batches = len(train_loader)

        pbar = tqdm(iterable=enumerate(train_loader), total=total_batches,
                    desc='Epoch {}/{}'.format(epoch, args.iter))
        loss_record = []
        for iteration, batch in pbar:
            v_i = batch[0]
            v_j = batch[1]

            loss = 0
            if args.order == 'all':
                for i in range(len(v_i)):
                    u_i1, u_j1, u_i2, u_j2 = model(node_index, v_i[i], v_j[i], device)
                    temp1 = torch.sum(torch.mul(u_i1, u_j1), dim=1)
                    temp2 = torch.sum(torch.mul(u_i2, u_j2), dim=1)
                    if i == 0:
                        loss1 = -torch.mean(F.logsigmoid(temp1), dim=0)
                        loss2 = -torch.mean(F.logsigmoid(temp2), dim=0)
                    else:
                        loss1 = -torch.mean(F.logsigmoid(-temp1), dim=0)
                        loss2 = -torch.mean(F.logsigmoid(-temp2), dim=0)
                    loss += (loss1 + loss2)
            else:
                for i in range(len(v_i)):
                    u_i, u_j = model(node_index, v_i[i], v_j[i], device)
                    temp = torch.sum(torch.mul(u_i, u_j), dim=1)
                    if i == 0:
                        temp = temp
                    else:
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
        embeddings = torch.mul(model.embeddings.weight.data, model.second_embeddings.weight.data).numpy()
    elif args.order == 'all':
        first_emb = model.embeddings.weight.data.numpy()
        second_emb = model.second_embeddings.weight.data.numpy()
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
