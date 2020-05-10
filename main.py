import numpy as np
import string
import argparse
import json
import csv

import sys

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support, f1_score

import torch
import torch.nn as nn

# import torch.nn as nn
# from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt

import gensim.downloader as api

np.random.seed(1234567)


def read_data(filename):
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, delimiter="\t")
        return [row for row in reader]


def w2vec(word_vecs, w):
    if "_" in w:  # a compound concept
        [w1, w2] = w.split("_")
        w1 = w1 if w1 in word_vecs else "unknown"
        w2 = w2 if w2 in word_vecs else "unknown"
        most_similar = word_vecs.most_similar_cosmul(positive=[w1, w2])[0]
        v = word_vecs[most_similar[0]]
        # word_vecs.add(w, v)
        return v
    elif w in word_vecs:
        return word_vecs.word_vec(w)
    # return the embedding for unknown word
    return word_vecs["unknown"]


class Model:
    def __init__(self, word_vecs, translation=False):
        self.wv = word_vecs
        self.translation = translation

        if translation:
            self.trans = torch.rand(200, requires_grad=True)
            self.model = lambda x: x + self.trans
        else:
            self.model = nn.Sequential(
                # Linear Transformation + bias is equivalent to an Affine Transformation
                nn.Linear(200, 200),
            )

    def train(self, words_train_set):

        # transform words to it embeddings
        embeddings_train_set = torch.tensor(
            list([w2vec(self.wv, h), w2vec(self.wv, t)] for [h, t] in words_train_set)
        )
        heads = embeddings_train_set[:, 0]
        tails = embeddings_train_set[:, 1]
        losses = []

        if self.translation:
            opt = optim.Adam([self.trans])
        else:
            opt = optim.Adam(self.model.parameters())

        criterion = torch.nn.MSELoss()

        for _epoch in range(2000):
            # a single epoch
            opt.zero_grad()

            y_hat = self.model(heads)
            # (3) Compute gradients
            loss = criterion(tails, y_hat)  # , torch.ones(tail.shape[0]))
            loss.backward()
            # (4) update weights
            opt.step()
            losses.append(loss.data.cpu().item())
        return losses

    def predict(self, head_words):
        head_embeddings = torch.tensor([w2vec(self.wv, h) for h in head_words])
        tail_words = []
        predicted_tes = self.model(head_embeddings).detach()
        for te in predicted_tes:
            predicted_tails = [
                w for (w, _score) in self.wv.similar_by_vector(te.numpy(), topn=20)
            ]
            tail_words.append(predicted_tails)
        return tail_words

    def test_precision_predict(self, words_dataset):
        # return the true tail word if it is near the predicted word
        hws = words_dataset[:, 0]
        tws = words_dataset[:, 1]
        predictions = self.predict(hws)

        res = []
        for near_predictions, tt in zip(predictions, tws):
            if tt in near_predictions:
                res.append(tt)
            else:
                res.append(near_predictions[0])
        return res

    def test_precision_truth(self, words_dataset):
        hws = words_dataset[:, 0]
        tws = words_dataset[:, 1]
        predictions = self.predict(hws)
        exact_predictions = [ps[0] for ps in predictions]

        res = []
        for exact_prediction, tt in zip(exact_predictions, tws):
            near_true = [w for (w, _score) in self.wv.most_similar(tt, topn=20)]
            if exact_prediction in near_true:
                res.append(tt)
            else:
                res.append(exact_prediction)
        return res

    def test_precision_both(self, words_dataset):
        # return tail if the the set of words near tail is not disjoint with
        # set of words near predicted tail
        hws = words_dataset[:, 0]
        tws = words_dataset[:, 1]
        predictions = self.predict(hws)

        res = []
        for near_predicted, tt in zip(predictions, tws):
            near_true = [w for (w, _score) in self.wv.most_similar(tt, topn=20)]
            if set(near_predicted).isdisjoint(set(near_true)):
                res.append(near_predicted[0])
            else:
                res.append(tt)
        return res


def visualize_tsne(words_strs, words_vecs):
    tsne_model_en_2d = TSNE(
        perplexity=50,
        n_components=2,
        # init="pca",
        n_iter=5000,
        random_state=32,
    )

    h_words_strs = words_strs[:, 0]
    t_words_strs = words_strs[:, 1]
    h_words_vecs = words_vecs[:, 0]
    t_words_vecs = words_vecs[:, 1]

    M, N, D = words_vecs.shape
    embeds_2d = torch.tensor(
        tsne_model_en_2d.fit_transform(np.concatenate([h_words_vecs, t_words_vecs]))
    ).view(M, 2, 2)
    h_2d = embeds_2d[:, 0]
    t_2d = embeds_2d[:, 1]
    plt.scatter(h_2d[:, 0], h_2d[:, 1], s=4)
    plt.scatter(t_2d[:, 0], t_2d[:, 1], s=4)

    for [xh, yh], [xt, yt] in zip(h_2d, t_2d):
        plt.arrow(xh, yh, xh - xt, yh - yt, alpha=0.1, head_width=0.2, head_length=0.2)

    def annotate_words(words, coords2d):
        for word, [_x, _y] in zip(words, coords2d):
            plt.annotate(
                word,
                alpha=0.3,
                xy=(_x, _y),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                size=8,
            )

    annotate_words(h_words_strs, h_2d)
    annotate_words(t_words_strs, t_2d)


def inverse(data):
    l = data[:, 0]
    r = data[:, 1]
    return np.array([r, l]).T


def run(data_file, word_vecs, translation=False, inverse=False):
    words_dataset = np.array(read_data(filename=data_file))

    words_dataset = np.array(
        [[h, t] for [h, t] in words_dataset if h in word_vecs and t in word_vecs]
    )

    if inverse:
        words_dataset = inverse(words_dataset)

    words_train, words_test = train_test_split(
        words_dataset, test_size=0.15, random_state=42,
    )

    relModel = Model(word_vecs, translation)
    relModel.train(words_train)

    word_predicted = relModel.test_precision_both(words_test)
    word_ground_truth = [tw for _hw, tw in words_test]

    precision, recall, fscore, _support = precision_recall_fscore_support(
        word_ground_truth, word_predicted, average="micro"
    )

    # for truth, pred in zip(words_test[:40], word_predicted[:40]):
    for truth, pred in zip(words_test, word_predicted):
        print(truth, pred)

    print("data_file:", data_file)
    print("translation:", translation)
    print("inverse:", inverse)
    print("precision: ", precision)
    print("recall: ", recall)
    print("fscore: ", fscore)
    print()


def runall(word_vecs):
    run("data/simple_IsA.pairs.csv", word_vecs, True, False)
    run("data/simple_IsA.pairs.csv", word_vecs, False, False)

    run("data/simple_CapableOf.pairs.csv", word_vecs, True, False)
    run("data/simple_CapableOf.pairs.csv", word_vecs, False, False)

    run("data/simple_Causes.pairs.csv", word_vecs, True, False)
    run("data/simple_Causes.pairs.csv", word_vecs, False, False)

    run("data/simple_UsedFor.pairs.csv", word_vecs, True, False)
    run("data/simple_UsedFor.pairs.csv", word_vecs, False, False)

    run("data/simple_Antonym.pairs.csv", word_vecs, False, False)
    run("data/simple_Antonym.pairs.csv", word_vecs, False, False)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runall", type=bool, default=False, help="runs all experiments",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="data/simple_CapableOf.pairs.csv",
        help="Dataset file",
    )

    parser.add_argument(
        "--inverse",
        type=bool,
        default=False,
        help="train for the inverse of the given relation",
    )

    parser.add_argument(
        "--translation",
        type=bool,
        default=False,
        help="if the model is just a translation. if false, then model is an affine transformation",
    )
    args = parser.parse_args()

    word_vecs = api.load("glove-wiki-gigaword-200")
    word_vecs.init_sims()

    if args.runall:
        runall(word_vecs)
    else:
        run(args.data_file, word_vecs, args.translation, args.inverse)

    plt.show()


if __name__ == "__main__":
    main()
