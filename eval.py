import math, random
#import gym
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from common.layers import NoisyLinear
from common.replay_buffer import ReplayBuffer

from rainbow import *
from envBuilder import *
from computeOptAction import *
from gen_dataset import *
from simulation import *

import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=np.int32, help="Number of nodes")
parser.add_argument("--ps", type=np.float32, help="Transition Probability")
parser.add_argument("--transitionModel", type=str, help="Transition Model")
parser.add_argument("--history", type=np.int32, help="History Level")

args = parser.parse_args()

n = args.n
ps = args.ps
transitionModel = args.transitionModel
history = args.history
env = envBuilder(n,ps,transitionModel,history)
num_input = len(env.reset())
print('num_input = ',num_input)
num_action = len(list(env.actionDict.keys()))

pathRL = './modelRL/'+str(n)+'Node/'+transitionModel+'/'+str(history)+'history/ps'+str(int(100*ps))+".pt"


basePath = './results/'+str(n)+'Node/'+transitionModel+'/'+str(history)+'history'+'/ps'+str(int(100*ps))+"/"
Path(basePath).mkdir(parents=True, exist_ok=True)

num_atoms = num_action
Vmin = 40
Vmax = 50

current_model = RainbowDQN(num_input, num_action, num_atoms, Vmin, Vmax)
# target_model  = RainbowDQN(num_input, num_action, num_atoms, Vmin, Vmax)

# if USE_CUDA:
#     current_model = current_model.cuda()
#     target_model  = target_model.cuda()
current_model.load_state_dict(torch.load(pathRL,map_location=torch.device('cpu')))
current_model.eval()

gamma = 1

# RF Model Load
print('pathRL = ',pathRL)


# ############################ test Code ##########################

testEpisodeCount = 500
rewardListRL = []
for k in range(testEpisodeCount):
    sRL = env.reset()
    done = False
    rewardRL = 0
    ind = 0

    actionListRL = []

    while not done:
        actRL = current_model.act(sRL)

        state,reward,done,_ = env.step(actRL)
        sRL = state
        actionListRL.append(actRL)

        rewardRL+=(gamma**ind)*reward

        ind+=1
    rewardListRL.append(rewardRL/env.countStepsMax)

print('Average reward (RL) = ',np.mean(rewardListRL))
# print('List of rewards stored at: ',basePath)
# np.savetxt(basePath+'outRewardSP.txt',rewardListSP,delimiter=',')



