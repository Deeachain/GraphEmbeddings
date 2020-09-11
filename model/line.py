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

from util.dataloader import read_graph, NodeDataLoader, get_alias_edge, alias_draw


class Line(nn.Module):
    def __init__(self, dict_size, embed_dim=128, order="first", num_negative=5):
        super(Line, self).__init__()

        assert order in ["first", "second", "all"], print("Order should either be [first, second, all]")

        self.embed_dim = embed_dim
        self.order = order
        self.first_embeddings = nn.Embedding(dict_size, embed_dim)
        self.second_embeddings = nn.Embedding(dict_size, embed_dim)
        self.num_negative = num_negative

        if order == "second":
            self.contextnodes_embeddings = nn.Embedding(dict_size, embed_dim)
            # Initialization
            self.contextnodes_embeddings.weight.data = self.contextnodes_embeddings.weight.data.uniform_(
                -1, 1) / embed_dim

        # Initialization
        self.first_embeddings.weight.data = self.first_embeddings.weight.data.uniform_(
            -1, 1) / embed_dim
        self.second_embeddings.weight.data = self.first_embeddings.weight.data.uniform_(
            -1, 1) / embed_dim

    def forward(self, v_i, v_j, negsamples, device):

        if self.order == 'first':
            v_i = self.first_embeddings(v_i).to(device)
            v_j = self.first_embeddings(v_j).to(device)


            mulpositivebatch = torch.mul(v_i, v_j)
            pos_loss = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

            neg_loss = 0
            for negativenodes in negsamples:
                negativeemb = -self.first_embeddings(negativenodes).to(device)
                mulnegativebatch = torch.mul(v_i, negativeemb)
                neg_loss += F.logsigmoid(-torch.sum(mulnegativebatch, dim=1))
            loss = pos_loss + neg_loss
        elif self.order == 'second':
            v_i = self.second_embeddings(v_i).to(device)
            v_j = self.contextnodes_embeddings(v_j).to(device)
            negativenodes = self.contextnodes_embeddings(negsamples).to(device)

            mulpositivebatch = torch.mul(v_i, v_j)
            pos_loss = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

            mulnegativebatch = torch.mul(v_i, negativenodes)
            neg_loss = F.logsigmoid(-torch.sum(mulnegativebatch, dim=1))
            loss = pos_loss + neg_loss
        elif self.order == 'all':
            v_i = self.second_embeddings(v_i).to(device)
            v_j1 = self.contextnodes_embeddings(v_j).to(device)
            negativenodes1 = -self.contextnodes_embeddings(negsamples).to(device)

            mulpositivebatch1 = torch.mul(v_i, v_j1)
            pos_loss1 = F.logsigmoid(torch.sum(mulpositivebatch1, dim=1))

            mulnegativebatch1 = torch.mul(v_i, negativenodes1)
            neg_loss1 = torch.sum(F.logsigmoid(torch.sum(mulnegativebatch1, dim=2)), dim=1)
            loss1 = pos_loss1 + neg_loss1

            v_i = self.second_embeddings(v_i).to(device)
            v_j2 = self.contextnodes_embeddings(v_j).to(device)
            negativenodes2 = -self.contextnodes_embeddings(negsamples).to(device)

            mulpositivebatch2 = torch.mul(v_i, v_j2)
            pos_loss2 = F.logsigmoid(torch.sum(mulpositivebatch2, dim=1))

            mulnegativebatch2 = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes2)
            neg_loss2 = torch.sum(F.logsigmoid(torch.sum(mulnegativebatch2, dim=2)), dim=1)
            loss2 = pos_loss2 + neg_loss2

            loss = loss1 + loss2

        return -torch.mean(loss)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    G = read_graph(args)
    nodes = sorted(list(G.nodes()))
    J_q, index2edge = get_alias_edge(G)  # alias table:(J,q)
    J, q = J_q[0], J_q[1]

    node2index = dict(zip(nodes, range(len(nodes))))
    dict_size = len(nodes)
    model = Line(dict_size, embed_dim=args.dimensions, order="first", num_negative=args.num_negative)

    NodeDataLoaderclass = NodeDataLoader(args=args, G=G, J=J, q=q, nodes=nodes,
                                         node2index=node2index, index2edge=index2edge)
    train_loader = NodeDataLoaderclass.TrainLoader()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    loss_record = []
    for epoch in range(args.iter):
        total_batches = len(train_loader)

        pbar = tqdm(iterable=enumerate(train_loader), total=total_batches,
                    desc='Epoch {}/{}'.format(epoch, args.iter))

        for iteration, batch in pbar:
            v_i = batch[0]
            v_j = batch[1]
            negsamples = batch[2]

            loss = model(v_i, v_j, negsamples, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(train_loss=float(loss))
            loss_record.append(loss)

    print('train loss is {}'.format(sum(loss_record) / len(loss_record)))

    if not os.path.exists(args.output_emb):
        os.makedirs(args.output_emb)
    emb_path = args.output_emb + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.emb'
    first_emb = model.first_embeddings.weight.data.numpy()
    second_emb = model.first_embeddings.weight.data

    index2node = dict(zip(range(len(nodes)), nodes))
    np.savetxt(emb_path, first_emb, fmt='%1.8f')

    with open(emb_path, 'r') as f1:
        lines = f1.readlines()
    with open(emb_path, 'w+') as f2:
        for i, l in enumerate(lines):
            f2.write('{} {}'.format(index2node[i], l))

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    checkpoints_path = args.checkpoints_path + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.pth'
    print("\nDone training, saving model to {}".format(checkpoints_path))
    torch.save(model, "{}".format(checkpoints_path))
