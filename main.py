# _*_ coding: utf-8 _*_
"""
Time:     2020/9/9 17:50
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     main.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
import os
import argparse
import numpy as np

from model import node2vec, deepwalk, line
from util.dataloader import read_graph
from util.word2vec import learn_embeddings
from util.evaluate import evaluate_embeddings
from util.visualization import plot_embeddings


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run deepwalk、line、node2vec.")
    parser.add_argument('--model_name', type=str, default='line',
                        help='Model choice in [deepwalk、line、node2vec]')
    parser.add_argument('--input', type=str, default='graph/cora/progressed/cora_edges.txt',
                        help='Input graph path')
    parser.add_argument('--input_label', type=str, default='graph/cora/progressed/cora_labels.txt',
                        help='Input graph path')
    parser.add_argument('--output_emb', type=str, default='output/embedding/',
                        help='Embeddings path')
    parser.add_argument('--output_pic', type=str, default='output/visualization/',
                        help='Plot_embeddings picture path')
    parser.add_argument('--checkpoints_path', type=str, default='output/checkpoints/',
                        help='checkpoint path of line')
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--num_negative', type=int, default=10,
                        help='Number of negativate sample. Default is 10.')
    parser.add_argument('--iter', default=500, type=int,
                        help='Number of epochs in SGD, Line Defalut should ')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batchsize for line')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimal')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=0.25,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=0.25,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--weighted', type=bool, default=False,
                        help='Graph is (un)weighted. Default is unweighted.')
    parser.add_argument('--directed', type=bool, default=False,
                        help='Graph is (un)directed. Default is undirected.')
    return parser.parse_args()


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph(args)

    if args.model_name == 'deepwalk':
        nx_G = deepwalk.load_edgelist(args.input, directed=args.directed)
        walks = deepwalk.build_deepwalk_corpus(nx_G, num_paths=args.num_walks,
                                               path_length=args.walk_length, alpha=0)
    elif args.model_name == 'line':
        line.main(args)
    elif args.model_name == 'node2vec':
        G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(args.num_walks, args.walk_length)

    print('Learning Embeddings...')
    if args.model_name == 'deepwalk' or args.model_name == 'node2vec':
        model = learn_embeddings(args, walks)

        if not os.path.exists(args.output_emb):
            os.makedirs(args.output_emb)
        emb_path = args.output_emb + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.emb'
        model.wv.save_word2vec_format(emb_path)

        embeddings = {}
        for node in nx_G.nodes():
            embeddings[str(node)] = model.wv[str(node)]
        args.log_file = args.output_pic + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.log'
        evaluate_embeddings(embeddings=embeddings, label_file=args.input_label, args=args)

        if not os.path.exists(args.output_pic):
            os.makedirs(args.output_pic)
        pic_path = args.output_pic + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.png'
        plot_embeddings(embeddings=embeddings, label_file=args.input_label, pic_path=pic_path)
    else:
        embeddings = {}
        emb_path = args.output_emb + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.emb'
        with open(emb_path, 'r') as f:
            lines = f.readlines()
        for l in lines:
            node = l.split(' ')[0]
            emb = l.split(' ')[1:]
            emb = np.array([float(i) for i in emb])
            embeddings[node] = emb

        args.log_file = args.output_pic + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.log'
        evaluate_embeddings(embeddings=embeddings, label_file=args.input_label, args=args)

        if not os.path.exists(args.output_pic):
            os.makedirs(args.output_pic)
        pic_path = args.output_pic + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.png'
        plot_embeddings(embeddings=embeddings, label_file=args.input_label, pic_path=pic_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
