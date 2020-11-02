# Adaptive Contention Window Design using Deep Q-learning

This repository is the official implementation of **"Adaptive Contention Window Design using Deep Q-learning"**. 

## Requirements

To install requirements:

```setup
docker pull pytorch/pytorch
```

>📋  (Optional) Install [NS3](https://www.nsnam.org/wiki/Installation) to generate new simulation dataset. Dataset used in the paper for training is provided in the **Dataset** directory.  

## Training

To train the model(s) in the paper, run this command:

```train
python trainRL.py  --n 10 --ps 1 --transitionModel Markovian --history 0
```

n: wireless network size; {5,10,20} 

ps: transition probability of stochastic process; {0.75,0.9,1}

transitionModel: stochastic model followed by other nodes; {'Markovian', 'NonMarkovian'} (Here 'NonMarkovian' corresponds to the 'complex' process in the paper.)

history: number of previous time steps from which observations are used as input; {0,1,2,3}

## Evaluation

To evaluate the model, run:

```eval
python eval.py --n 10 --ps 1 --transitionModel Markovian --history 0
```

## Pre-trained Models

>📋  Pretrained models are present in **modelRL** directory.
