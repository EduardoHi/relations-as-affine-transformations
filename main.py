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


def train(embeddings_dataset, words_train_set, word_vecs):
    word_ground_truth = [tw for _hw, tw in words_train_set]

    trans = torch.rand(200, requires_grad=True)
    opt = optim.Adam([trans])

    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CosineEmbeddingLoss()

    losses = []

    head = embeddings_dataset[:, 0]
    tail = embeddings_dataset[:, 1]

    for epoch in range(5000):
        # a single epoch
        opt.zero_grad()

        # a simple translation
        y_hat = head + trans

        loss = criterion(tail, y_hat)  # , torch.ones(tail.shape[0]))

        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.cpu().item())

    trans = trans.detach()
    torch.save(trans, "trans.pt")

    plt.plot(losses)

    return trans, losses


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


def predict(trans, head_words, word_vecs):
    head_embeddings = torch.tensor([w2vec(word_vecs, h) for h in head_words])
    tail_words = []
    for he in head_embeddings:
        predicted_te = he + trans

        predicted_tails = [
            w
            for (w, _score) in word_vecs.similar_by_vector(
                predicted_te.detach().numpy()
            )
        ]
        tail_words.append(predicted_tails)
    return tail_words


def test_predict(trans, words_dataset, word_vecs):
    hws = words_dataset[:, 0]
    tws = words_dataset[:, 1]
    predictions = predict(trans, hws, word_vecs)

    def f(pair):
        tws_hat, true_tw = pair
        if true_tw in tws_hat:
            return true_tw
        else:
            return tws_hat[0]

    return list(map(f, zip(predictions, tws)))

    word_predicted = []
    for (hw, tw), (he, te) in zip(words_dataset, embeds_dataset):
        te_hat = he + trans
        tw_hats = [
            w for (w, score) in word_vecs.similar_by_vector(te_hat.detach().numpy())
        ]
        # a little help, it's correct if it is in the top-10 similar vectors
        if tw in tw_hats:
            word_predicted.append(tw)
        else:
            word_predicted.append(tw_hats[0])

    return word_predicted


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/simple_CapableOf.pairs.csv",
        help="Dataset file",
    )
    parser.add_argument(
        "--vecs_file",
        type=str,
        default="data/CapableOf.vecs.pt",
        help="word vectors file",
    )
    parser.add_argument(
        "--epochs", type=int, default=2000,
    )
    args = parser.parse_args()

    words_dataset = np.array(read_data(filename=args.data_file))

    word_vecs = api.load("glove-wiki-gigaword-200")
    word_vecs.init_sims()

    embeddings_dataset = torch.tensor(
        list([w2vec(word_vecs, h), w2vec(word_vecs, t)] for [h, t] in words_dataset)
    )
    # (M , 2, 200)
    # M is number of samples, 2 since it's a pair, and 200 the dimensions of the embeddings

    print("loaded data and word_vecs", datetime.now().time(), file=sys.stderr)

    # visualize_tsne(words_dataset, embeddings_dataset)

    words_train, words_test, embeds_train, embeds_test = train_test_split(
        words_dataset, embeddings_dataset.numpy(), test_size=0.20, random_state=42,
    )

    trans, losses = train(torch.tensor(embeds_train), words_train, word_vecs)

    print("finished training", datetime.now().time(), file=sys.stderr)

    # predict against the whole dataset, not train and test splits
    word_predicted = test_predict(trans, words_dataset, word_vecs)
    print(word_predicted)
    word_ground_truth = [tw for _hw, tw in words_dataset]

    precision, recall, fscore, _support = precision_recall_fscore_support(
        word_ground_truth, word_predicted, average="micro"
    )

    for truth, pred in zip(word_ground_truth, word_predicted):
        print(truth, pred)

    print("precision: ", precision)
    print("recall: ", recall)
    print("fscore: ", fscore)

    # visualize_tsne(words_test, embeds_test)
    plt.show()


if __name__ == "__main__":
    main()
