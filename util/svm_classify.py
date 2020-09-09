# _*_ coding: utf-8 _*_
"""
Time:     2020/9/9 17:50
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     ding.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Deeachain/GraphEmbeddings
"""
import numpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC


class Classifier(object):

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.clf = SVC(kernel='rbf')  # 'rbf','poly','sigmoid', 'linear'

    def train(self, X, Y):
        X_train = [self.embeddings[x] for x in X]

        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        Y_ = self.predict(X)
        averages = ["micro", "macro", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y, Y_)
        print('Evaluating classifier using rest nodes...')
        print('-------------------')
        print(results)
        print('-------------------')
        return results

    def predict(self, X):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_)
        Y = numpy.asarray(Y)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
        self.train(X_train, Y_train)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)
