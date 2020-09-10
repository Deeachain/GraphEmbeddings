# _*_ coding: utf-8 _*_
"""
Time:     2020/9/10 10:47
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     line.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from util.dataloader import read_graph, NodeDataLoader


class Line(nn.Module):
    def __init__(self, dict_size, embed_dim=128, order="first"):
        super(Line, self).__init__()

        assert order in ["first", "second", "all"], print("Order should either be [first, second, all]")

        self.embed_dim = embed_dim
        self.order = order
        self.first_embeddings = nn.Embedding(dict_size, embed_dim)
        self.second_embeddings = nn.Embedding(dict_size, embed_dim)

        if order == "second":
            self.contextnodes_embeddings = nn.Embedding(dict_size, embed_dim)
            # Initialization
            self.contextnodes_embeddings.weight.data = self.contextnodes_embeddings.weight.data.uniform_(
                -.5, .5) / embed_dim

        # Initialization
        self.first_embeddings.weight.data = self.first_embeddings.weight.data.uniform_(
            -.5, .5) / embed_dim
        self.second_embeddings.weight.data = self.first_embeddings.weight.data.uniform_(
            -.5, .5) / embed_dim

    def forward(self, v_i, v_j, negsamples, device):
        if self.order == 'first':
            v_i = self.first_embeddings(v_i).to(device)
            v_j = self.first_embeddings(v_j).to(device)
            negativenodes = -self.first_embeddings(negsamples).to(device)

            mulpositivebatch = torch.mul(v_i, v_j)
            positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

            mulnegativebatch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes)
            negativebatch = torch.sum(F.logsigmoid(torch.sum(mulnegativebatch, dim=2)), dim=1)
            loss = positivebatch + negativebatch
        elif self.order == 'second':
            v_i = self.second_embeddings(v_i).to(device)
            v_j = self.contextnodes_embeddings(v_j).to(device)
            negativenodes = -self.contextnodes_embeddings(negsamples).to(device)

            mulpositivebatch = torch.mul(v_i, v_j)
            positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

            mulnegativebatch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes)
            negativebatch = torch.sum(F.logsigmoid(torch.sum(mulnegativebatch, dim=2)), dim=1)
            loss = positivebatch + negativebatch
        elif self.order == 'all':
            v_i = self.second_embeddings(v_i).to(device)
            v_j1 = self.contextnodes_embeddings(v_j).to(device)
            negativenodes1 = -self.contextnodes_embeddings(negsamples).to(device)

            mulpositivebatch1 = torch.mul(v_i, v_j1)
            positivebatch1 = F.logsigmoid(torch.sum(mulpositivebatch1, dim=1))

            mulnegativebatch1 = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes1)
            negativebatch1 = torch.sum(F.logsigmoid(torch.sum(mulnegativebatch1, dim=2)), dim=1)
            loss1 = positivebatch1 + negativebatch1

            v_i = self.second_embeddings(v_i).to(device)
            v_j2 = self.contextnodes_embeddings(v_j).to(device)
            negativenodes2 = -self.contextnodes_embeddings(negsamples).to(device)

            mulpositivebatch2 = torch.mul(v_i, v_j2)
            positivebatch2 = F.logsigmoid(torch.sum(mulpositivebatch2, dim=1))

            mulnegativebatch2 = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes2)
            negativebatch2 = torch.sum(F.logsigmoid(torch.sum(mulnegativebatch2, dim=2)), dim=1)
            loss2 = positivebatch2 + negativebatch2

            loss = loss1 + loss2

        return -torch.mean(loss)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    G = read_graph(args)
    nodes = sorted(list(G.nodes()))
    dict_size = len(nodes)
    model = Line(dict_size, embed_dim=args.dimensions, order="first")

    NodeDataLoaderclass = NodeDataLoader(args)
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
            loss_record.append(loss)

    print('train loss is {}'.format(sum(loss_record)/len(loss_record)))

    if not os.path.exists(args.output_emb):
        os.makedirs(args.output_emb)
    emb_path = args.output_emb + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.emb'
    first_emb = model.first_embeddings.weight.data
    second_emb = model.first_embeddings.weight.data
    first_emb = first_emb.numpy()
    with open(emb_path, 'w+') as f:
        f.write(first_emb)

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    checkpoints_path = args.checkpoints_path + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.pth'
    print("\nDone training, saving model to {}".format(checkpoints_path))
    torch.save(model, "{}".format(checkpoints_path))
