# Report associated to this project is in folder written

to run all evaluations, do:

```sh
$ python3 main.py --runall
```

for a specific dataset:

```sh
$ python3 main.py --data_file <data file> --translation <True or False> --inverse <True or False>
```

`main.py` is a self-contained file with both the model
and the cli. other `*.py` included are just to preprocess data.

# ConceptNet Data Attribution

This work includes data from ConceptNet 5,
which was compiled by the Commonsense Computing Initiative.
ConceptNet 5 is freely available under the Creative Commons
Attribution-ShareAlike license (CC BY SA 4.0) from http://conceptnet.io.
The included data was created by contributors to Commonsense Computing
projects, contributors to Wikimedia projects, Games with a Purpose,
Princeton University's WordNet, DBPedia, OpenCyc, and Umbel.
