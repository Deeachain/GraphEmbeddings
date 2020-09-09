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

from model import node2vec, deepwalk
from util.dataloader import read_graph
from gensim.models import Word2Vec
from util.evaluate import evaluate_embeddings
from util.visualization import plot_embeddings


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)

    return model


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph(args)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)

    print('Learning Embeddings...')
    model = learn_embeddings(walks)

    if not os.path.exists(args.output_emb):
        os.makedirs(args.output_emb)
    emb_path = args.output_emb + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.emb'
    model.wv.save_word2vec_format(emb_path)

    embeddings = {}
    for node in nx_G.nodes():
        embeddings[str(node)] = model.wv[str(node)]

    evaluate_embeddings(embeddings=embeddings, label_file=args.input_label)

    if not os.path.exists(args.output_pic):
        os.makedirs(args.output_pic)
    pic_path = args.output_pic + args.model_name + '_' + args.input.split('/')[-1].split('.')[0] + '.png'
    plot_embeddings(embeddings=embeddings, label_file=args.input_label, pic_path=pic_path)


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run deepwalk、line、node2vec.")
    parser.add_argument('--model_name', type=str, default='node2vec',
                        help='Model choice in [deepwalk、line、node2vec]')
    parser.add_argument('--input', type=str, default='graph/cora/progressed/cora_edges.txt',
                        help='Input graph path')
    parser.add_argument('--input_label', type=str, default='graph/cora/progressed/cora_labels.txt',
                        help='Input graph path')
    parser.add_argument('--output_emb', type=str, default='output/out_embedding/',
                        help='Embeddings path')
    parser.add_argument('--output_pic', type=str, default='output/visualization/',
                        help='Plot_embeddings picture path')
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
