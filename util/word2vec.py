# _*_ coding: utf-8 _*_
"""
Time:     2020/9/9 19:58
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     word2vec.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
from gensim.models import Word2Vec


def learn_embeddings(args, walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)

    return model
