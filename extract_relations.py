import csv
import json
import sys

from uri_utils import *

# print the extracted triplets to a stdout as csv with tabs
def out_triplets(triplets):
    with sys.stdout as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        for triplet in triplets:
            rel, head, tail, weight = triplet
            writer.writerow([join_uri(*rel), join_uri(*head), join_uri(*tail), weight])


# generator of every triplet: (relation, concept, concept, weight)
# it actually is not a triplet because it has four elements, the weight
# but really weight is not a standalone concept of the triplet but related to
# the triplet as a whole
def gen_triplets():
    with sys.stdin as csv_file:
        reader = csv.reader(csv_file, delimiter="\t")
        for row in reader:
            yield (
                split_uri(row[0]),
                split_uri(row[1]),
                split_uri(row[2]),
                row[3],
                # json.loads(row[4])["weight"],
            )


# concept uri shape: c/<lang>/<word>
# relation uri shape: r/<relation>


def filter_uninteresting_relations(triplets):
    for triplet in triplets:
        relation, head_concept, tail_concept, _ = triplet
        _, rel, *_ = relation
        if rel not in {
            "ExternalURL",
            # deprecated relations
            "dbpedia",
            "InstanceOf",
            "Entails",
        }:
            yield triplet


def filter_both_en(triplets):
    for triplet in triplets:
        try:
            relation, head_concept, tail_concept, _ = triplet
            _, hlang, *_ = head_concept
            _, tlang, *_ = tail_concept
        except ValueError:
            print(triplet, file=sys.stderr)
            raise ValueError
        if hlang == "en" and tlang == "en":
            yield triplet


def filter_compound_concepts(triplets):
    for triplet in triplets:
        relation, head_concept, tail_concept, _ = triplet
        _, _, hconcept, *_ = head_concept
        _, _, tconcept, *_ = tail_concept
        if len(hconcept.split("_")) <= 2 and len(tconcept.split("_")) <= 2:
            yield triplet


class Relation:
    def __init__(self, name, pairs):
        self.name = name
        self.pairs = pairs

    def arity_stats(self):
        """ returns the max number of edges from head to tail and from tail to head"""
        # """ returns either "1-1", "1-M", "N-1", "N-M" """
        l = dict()
        r = dict()

        for (h, t) in self.pairs:
            if h in l:
                l[h] += 1
            else:
                l[h] = 1
            if t in r:
                r[t] += 1
            else:
                r[t] = 1

        lmax = max(l.values())
        rmax = max(r.values())

        # print("right to left")
        mkr = ""
        for k in r:
            # print(k, r[k])
            mkr = k if r[k] == rmax else mkr

        print("left to right")
        mkl = ""
        for k in l:
            print(k, l[k])
            mkl = k if l[k] == lmax else mkl

        print("max")
        print(mkl, "-> _", lmax)
        print(rmax, "_ <-", mkr)

        lavg = sum(l.values()) / len(l.values())
        ravg = sum(r.values()) / len(r.values())
        print("avg", lavg, ravg)

    def relation_to_csv(self):
        with sys.stdout as csv_file:
            writer = csv.writer(csv_file, delimiter="\t")
            for p in self.pairs:
                writer.writerow(p)


def triplets_to_relation(triplets):
    triplets = list(triplets)
    names = set((r for (_, r, *_), *_ in triplets))
    if len(names) > 1:
        raise ValueError("Expected triplets from only one relation")

    name = names.pop()

    pairs = set((h, t) for _, (_, _, h, *_), (_, _, t, *_), _ in triplets)

    return Relation(name, pairs)


if __name__ == "__main__":
    triplets = gen_triplets()
    r = triplets_to_relation(triplets)
    # r.arity_stats()
    r.relation_to_csv()

    # triplets = filter_uninteresting_relations(triplets)
    # # triplets = filter_both_en(triplets)
    # triplets = filter_compound_concepts(triplets)
    # out_triplets(triplets)
