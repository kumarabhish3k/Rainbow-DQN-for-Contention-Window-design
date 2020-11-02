# Adaptive Contention Window Design using Deep Q-learning

This repository is the official implementation of [Adaptive Contention Window Design using Deep Q-learning
]. 

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

## Evaluation

To evaluate my model, run:

```eval
python eval.py --n 10 --ps 1 --transitionModel Markovian --history 0
```

## Pre-trained Models

>📋  Pretrained models are present in modelRL directory.
