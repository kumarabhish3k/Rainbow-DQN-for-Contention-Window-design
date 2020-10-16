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
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class RainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowDQN, self).__init__()
        
        self.num_inputs   = num_inputs
        self.num_actions  = num_actions
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax
        # Markovian
        numNodes = 16
        # NonMarkovian
        # numNodes = 48

        self.linear1 = nn.Linear(num_inputs, numNodes)
        self.linear2 = nn.Linear(numNodes, numNodes)
        
        # numNodes = 16

        self.noisy_value1 = NoisyLinear(numNodes, numNodes, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(numNodes, self.num_atoms, use_cuda=USE_CUDA)
        
        self.noisy_advantage1 = NoisyLinear(numNodes, numNodes, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(numNodes, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
      
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        
        value     = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)
        
        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
       
        return x
        
    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()
    
    def act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action



def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def projection_distribution(next_state, rewards, dones,Vmax,Vmin,num_atoms,target_model,batch_size):
    batch_size  = next_state.size(0)
    
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms)
    
    next_dist   = target_model(next_state).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist   = next_dist.gather(1, next_action).squeeze(1)
        
    rewards = rewards.unsqueeze(1).expand_as(next_dist)
    dones   = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)
    
    Tz = rewards + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b  = (Tz - Vmin) / delta_z
    l  = b.floor().long()
    u  = b.ceil().long()
        
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, num_atoms)

    proj_dist = torch.zeros(next_dist.size())    
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        
    return proj_dist

def compute_td_loss(batch_size,replay_buffer,Vmax,Vmin,num_atoms,current_model,target_model,optimizer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size) 
    #print('Vmax = ',Vmax)
 
    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(np.float32(done))
    #print(state)
    proj_dist = projection_distribution(next_state, reward, done,Vmax,Vmin,num_atoms,target_model,batch_size)
    
    dist = current_model(state)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, action).squeeze(1)
    dist.data.clamp_(0.01, 0.99)
    loss = -(Variable(proj_dist) * dist.log()).sum(1)
    loss  = loss.mean()
    #print(loss)    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss

if __name__ == "__main__":
    path = './model/modelps1.pt'
    ps = 1
    n = 10
    dict1 = gen_data(n)

    num_input = 3
    
    action_dict = dict()
    actions = [32,48,64,80,96,112,128]
    for i in range(len(actions)):
        action_dict[i]=actions[i]

    other_action_dict = dict()
    other_action_dict[0] = 32
    other_action_dict[1] = 128

    num_states = 3
    num_action = len(actions)
    eplen = 20

    env = commEnv(ps,dict1,action_dict,other_action_dict,eplen)
 

    # env = commEnv(ps,dict1,action_dict)


    num_atoms = 5
    Vmin = 16
    Vmax = 20

    current_model = RainbowDQN(num_input, num_action, num_atoms, Vmin, Vmax)
    target_model  = RainbowDQN(num_input, num_action, num_atoms, Vmin, Vmax)

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()
    
    optimizer = optim.Adam(current_model.parameters(), 0.0001)

    replay_buffer = ReplayBuffer(10000)

    update_target(current_model, target_model)

    num_frames = 10000
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
        replay_buffer.push(state, action, reward, next_state, done)
        
        #print(reward)
        state = next_state
        #print(cw_min)
        episode_reward += reward
        if episode_reward > 195:
            break 
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