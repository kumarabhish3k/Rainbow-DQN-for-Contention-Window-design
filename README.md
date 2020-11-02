# Adaptive Contention Window Design using Deep Q-learning

This repository is the official implementation of [Adaptive Contention Window Design using Deep Q-learning
]. 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
docker pull pytorch/pytorch
```

>📋  (Optional) Install [NS3](https://www.nsnam.org/wiki/Installation) to generate new simulation dataset. 

## Training

To train the model(s) in the paper, run this command:

```train
python trainRL.py  --n 10 --ps 1 --transitionModel Markovian --history 0
```

>📋  For 

## Evaluation

To evaluate my model, run:

```eval
python eval.py --n 10 --ps 1 --transitionModel Markovian --history 0
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

>📋  Pretrained models are present in modelRL directory.
