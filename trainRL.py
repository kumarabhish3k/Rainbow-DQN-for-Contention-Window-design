import math, random
from simulation import *
#import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from common.layers import NoisyLinear
from common.replay_buffer import ReplayBuffer
from gen_dataset import *

# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
from rainbow import *
from envBuilder import *
import argparse
from pathlib import Path

#parser = argparse.ArgumentParser()
#parser.add_argument("--n", type=np.int32, help="Number of nodes")
#parser.add_argument("--ps", type=np.float32, help="Transition Probability")
#parser.add_argument("--transitionModel", type=str, help="Transition Model")
#parser.add_argument("--history", type=np.int32, help="History Level")
#args = parser.parse_args()

# n = args.n
# ps = args.ps
# transitionModel = args.transitionModel
# history = args.history

n = 5
ps = 1
transitionModel = 'Markovian'
history = 0


env = envBuilder(n,ps,transitionModel,history)


num_input = len(env.reset()) ##state dimension
print('Input Dimension = ',num_input)
num_action = len(list(env.actionDict.keys()))
# print('num_action',num_action)

baseFolder ='./modelRL/'+str(n)+'Node/'+transitionModel+'/'+str(history)+'history/' 
Path(baseFolder).mkdir(parents=True, exist_ok=True)
path = baseFolder+'ps'+str(int(100*ps))+".pt"

num_atoms = num_action
Vmin = 45 #40
Vmax = 50

current_model = RainbowDQN(num_input, num_action, num_atoms, Vmin, Vmax)
target_model  = RainbowDQN(num_input, num_action, num_atoms, Vmin, Vmax)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()

optimizer = optim.Adam(current_model.parameters(), 0.0002)

replay_buffer = ReplayBuffer(15000)

update_target(current_model, target_model)

num_frames = 30000
batch_size = 32
gamma      = 1

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
# print(state)
for frame_idx in range(1, num_frames + 1):

    action = current_model.act(state)
#       print('action = ',action)

    next_state, reward, done, _= env.step(action)
    print("frame_idx: "+str(frame_idx))
    print("state: " + str(state) )
    print("action: " + str(env.actionDict[action]))
    print("reward: "+str(reward))
    replay_buffer.push(state, action, reward, next_state, done)
    
    #print(reward)
    state = next_state
    #print(cw_min)
    episode_reward += reward
    if done:
        print('Frame = ',frame_idx,'Reward = ',episode_reward)
        state= env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
    
    if len(replay_buffer) > batch_size:
        #print('======================================= Start Training =========================================')
        loss = compute_td_loss(batch_size,replay_buffer,Vmax,Vmin,num_atoms,current_model,target_model,optimizer)
        #print(loss)
        losses.append(loss.data)
    #if frame_idx % 200 == 0:
    #    plot(frame_idx, all_rewards, losses)
    
    if frame_idx % 500 == 0:
        update_target(current_model, target_model)
#print(all_rewards)
torch.save(current_model.state_dict(),path)






