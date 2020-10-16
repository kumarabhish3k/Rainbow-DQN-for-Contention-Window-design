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
from joblib import dump, load

from rainbow import *
from envBuilder import *
from computeOptAction import *
from gen_dataset import *
from simulation import *

import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("n", type=np.int32, help="Number of nodes")
parser.add_argument("ps", type=np.float32, help="Transition Probability")
parser.add_argument("transitionModel", type=str, help="Transition Model")
parser.add_argument("history", type=np.int32, help="History Level")

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
pathRF = './modelRF/'+str(n)+'Node/'+transitionModel+'/'+str(history)+'history/ps'+str(int(100*ps))+"/"

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
clf = load(pathRF+'rf.joblib')

# ############################ test Code ##########################

testEpisodeCount = 500
rewardListRL = []
rewardListOpt = []
rewardListCP = []
rewardListRF = []

for k in range(testEpisodeCount):
    sRL = env.reset()
    sRF  = np.copy(sRL)
    print(sRL)
    done = False
    rewardRL = 0
    rewardOpt = 0
    rewardRF = 0
    rewardCP = 0
    ind = 0

    actionListRL = []
    actionListRF = []
    actionListOpt = []
    actionListCP = []
    actionListTrue = []
    while not done:
        actRL = current_model.act(sRL)
        # Optimal action
        actOpt = computeOptAction(env)
        # pdb.set_trace()
        actRF = clf.predict(sRF.reshape(1,-1))[0]
        # print('RF Action  = ',actRF)
        # Get index in actionDict
        cwRF = env.otherActionDict[actRF]
        actRF = list(env.actionDict.keys())[list(env.actionDict.values()).index(cwRF)]

        if transitionModel == "Markovian":
            actCP = 4
        elif transitionModel == "NonMarkovian":
            actCP = 6

        actionList = [actRL,actOpt,actCP,actRF]
    
        stateList,rewardList,done,_ = env.stepTest(actionList)
        sRL = stateList[0]
        sRF = stateList[-1]
        actionListTrue.append(env.otherActionIndex)
        actionListRL.append(actRL)
        actionListRF.append(actRF)
        actionListOpt.append(actOpt)
        actionListCP.append(actCP)
        # print('RL Action  = ',actRL)
        # print('True Action = ',env.otherActionIndex)
        # print('Correct action = ',env.otherActionIndex)
        # print('Predicted Action = ',action_dict[actRL])
        print('Optimal Action = ',env.actionDict[actOpt])
        # print('Correct Action = ',other_action_dict[env.otherActionIndex])

        # if (action_dict[actRL]== other_action_dict[env.otherActionIndex]):
        #     count +=1
        # print('RL Reward = ',rewardList[0])
        # print('Opt Reward = ',rewardList[1])
        # print('RF Reward = ',rewardList[2])

        rewardRL+=(gamma**ind)*rewardList[0]
        rewardOpt+=(gamma**ind)*rewardList[1]
        rewardCP+=(gamma**ind)*rewardList[2]
        rewardRF+=(gamma**ind)*rewardList[3]

        ind+=1
    rewardListRL.append(rewardRL)
    rewardListOpt.append(rewardOpt)
    rewardListCP.append(rewardCP)
    rewardListRF.append(rewardRF)
print('Average reward (RL) = ',np.mean(rewardListRL))
print('Average reward (Opt) = ',np.mean(rewardListOpt))
print('Average reward (CP) = ',np.mean(rewardListCP))
print('Average reward (RF) = ',np.mean(rewardListRF))



# np.savetxt(basePath+'outRewardRL.txt',rewardListRL,delimiter=',')
# np.savetxt(basePath+'outRewardOpt.txt',rewardListOpt,delimiter=',')
# np.savetxt(basePath+'outRewardCP.txt',rewardListCP,delimiter=',')
# np.savetxt(basePath+'outRewardRF.txt',rewardListRF,delimiter=',')



