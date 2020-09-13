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

def Joint_probability(v_i_embedding, v_j_embedding, negative_sample_embedding):
    mulpositivebatch = torch.sum(torch.mul(v_i_embedding, v_j_embedding), dim=1)
    pos_loss = F.logsigmoid(mulpositivebatch)

    neg_loss = 0
    for negative_embedding in negative_sample_embedding:
        mulnegativebatch = torch.sum(torch.mul(v_i_embedding, negative_embedding), dim=1)
        neg_loss += F.logsigmoid(-mulnegativebatch)

    loss = pos_loss + neg_loss
    return loss


class Line(nn.Module):
    def __init__(self, dict_size, embed_dim=128, order="first", num_negative=5):
        super(Line, self).__init__()

        assert order in ["first", "second", "all"], print("Order should either be [first, second, all]")
        self.dict_size = dict_size
        self.embed_dim = embed_dim
        self.order = order
        self.first_embeddings = nn.Embedding(dict_size, embed_dim)
        self.second_embeddings = nn.Embedding(dict_size, embed_dim)
        self.context_embeddings = nn.Embedding(dict_size, embed_dim)
        self.num_negative = num_negative


    def forward(self, nodeindex, v_i, v_j, negsamples, device):
        # input
        v_i = torch.LongTensor(v_i)
        v_j = torch.LongTensor(v_j)
        negsamples = [torch.LongTensor(negative) for negative in negsamples]
        # onehot
        v_i = torch.Tensor(np.eye(self.dict_size)[v_i]).to(device)
        v_j = torch.Tensor(np.eye(self.dict_size)[v_j]).to(device)
        negative_sample_onehot = [torch.Tensor(np.eye(self.dict_size)[negative]).to(device) for negative in negsamples]
        # init embeddings
        first_embeddings = self.first_embeddings(torch.LongTensor(nodeindex)).to(device)
        second_embeddings = self.second_embeddings(torch.LongTensor(nodeindex)).to(device)
        context_embeddings = self.context_embeddings(torch.LongTensor(nodeindex)).to(device)

        if self.order == 'first':
            v_i_embedding = torch.mm(v_i, first_embeddings)
            v_j_embedding = torch.mm(v_j, first_embeddings)
            negative_sample_embedding = [torch.mm(negative, first_embeddings) for negative in negative_sample_onehot]

            loss = Joint_probability(v_i_embedding, v_j_embedding, negative_sample_embedding)
        elif self.order == 'second':
            v_i_embedding = torch.mm(v_i, second_embeddings)
            v_j_embedding = torch.mm(v_j, context_embeddings)
            negative_sample_embedding = [torch.mm(negative, context_embeddings) for negative in negative_sample_onehot]

            loss = Joint_probability(v_i_embedding, v_j_embedding, negative_sample_embedding)
        elif self.order == 'all':
            v_i_embedding1 = torch.mm(v_i, first_embeddings)
            v_j_embedding1 = torch.mm(v_j, first_embeddings)
            negative_sample_embedding1 = [torch.mm(negative, first_embeddings) for negative in negative_sample_onehot]

            v_i_embedding2 = torch.mm(v_i, second_embeddings)
            v_j_embedding2 = torch.mm(v_j, context_embeddings)
            negative_sample_embedding2 = [torch.mm(negative, context_embeddings) for negative in negative_sample_onehot]

            loss1 = Joint_probability(v_i_embedding1, v_j_embedding1, negative_sample_embedding1)
            loss2 = Joint_probability(v_i_embedding2, v_j_embedding2, negative_sample_embedding2)
            loss = loss1 + loss2
        return -torch.mean(loss)


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
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    for epoch in range(args.iter):
        total_batches = len(train_loader)

        pbar = tqdm(iterable=enumerate(train_loader), total=total_batches,
                    desc='Epoch {}/{}'.format(epoch, args.iter))
        loss_record = []
        for iteration, batch in pbar:
            v_i = batch[0]
            v_j = batch[1]

            negsamples = batch[2]

            loss = model(node_index, v_i, v_j, negsamples, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(train_loss=float(loss))
            loss_record.append(loss)

    print('train loss is {}'.format(sum(loss_record) / len(loss_record)))

    if args.order == 'first':
        embeddings = model.first_embeddings.weight.data.numpy()
    elif args.order == 'second':
        embeddings = model.second_embeddings.weight.data.numpy()
    elif args.order == 'all':
        first_emb = model.first_embeddings.weight.data.numpy()
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
