import numpy as np
import os
import random
from math import *
from gen_dataset import *
# from envBuilder import *

class commEnv:
    def __init__(self,n,ps,transitionModel,action_dict,other_action_dict,eplen,history=0):
        """
        args: T     - Simulation time in seconds
              N     - Number of nodes in the network
              cwMin - Minimum contention window duration
              cwMax - maximum contention window duration
              ifs   - Interframe space
              L     - maximum length of message
        
        """
        self.n = n
        self.transitionModel = transitionModel
        self.ps = ps    
        self.countStepsMax = eplen
        self.actionDict = action_dict
        self.otherActionDict = other_action_dict

        self.dimState = 3
        self.numTestMethod = 4
        self.dict = genDataset(n)
        baseFolder = './Dataset/dataStats/'+str(n)+'Node/'
        self.data_mean = np.loadtxt(baseFolder+'data_mean.txt',delimiter = ',')
        self.data_std = np.loadtxt(baseFolder+'data_std.txt',delimiter = ',')
        # Count for no. of steps in one episode
        self.countSteps = 0
        self.otherActionIndex = random.choice(list(self.otherActionDict.keys()))
        self.incrementFlag = random.choice([True, False])

        self.historyFlag = history

        if self.historyFlag:
            self.historyData = []
            self.historyTestData = []


    def preProcess(self,data):
        data1 = np.divide(data-self.data_mean,self.data_std)
        # data_temp = data[:,:-1]
        # data_vec = data_temp.flatten('F');
        # rew = data[0,-1]
        # data_vec = np.append(data_vec,rew)
        return data1
    
    def computeReward(self,rhoOmegaDiff):
        reward = (1-rhoOmegaDiff)
        return reward 


    def reset(self):
        self.countSteps = 0 
        self.otherActionIndex = random.choice(list(self.otherActionDict.keys()))
        stRaw = random.choice(self.dict[str(random.choice(self.actionDict))+'+'+str(self.otherActionDict[self.otherActionIndex])])
        stNormalized = self.preProcess(np.asarray(stRaw[0:self.dimState]))

        if self.historyFlag:
            self.historyData = np.tile(stNormalized,self.historyFlag)
            st = np.concatenate([stNormalized,self.historyData],axis=0)

            # Test Data
            for i in range(self.numTestMethod):
                self.historyTestData.append(self.historyData)
        else:
            st = stNormalized
        return st
     
    def step(self,a):
        self.countSteps+=1
        self.change_env_state()
        # print('Key = ',self.otherActionIndex)
        key = str(self.actionDict[a])+'+'+str(self.otherActionDict[self.otherActionIndex])

        next_state_full = random.choice(self.dict[key])        
        next_state_normalized = self.preProcess(np.asarray(next_state_full[0:self.dimState]))
        
        done = False
        reward1= self.computeReward(next_state_full[-1])
        info = 0
        """
        if close:
            if self.countIterationflag:
                self.countIteration +=1
            else:
                self.countIterationflag = True
                self.countIteration +=1
        else:
            self.countIterationflag = False
            self.countIteration = 0
       
        if close:
            reward1+=self.count

        if (self.countIteration == self.countIterationMax):
            print('Model Converged')
            done = True
            self.countIteration = 0
            self.countIterationflag = False
        """
        if (self.countSteps >= self.countStepsMax):
            done = True
            #self.countIteration = 0
            #self.countIterationflag = False
            self.countSteps = 0

        if self.historyFlag:
            next_state = np.concatenate([next_state_normalized,self.historyData],axis=0)
            
            # Append current observation to history
            self.historyData = np.roll(self.historyData,self.dimState)
            self.historyData[0:self.dimState] = np.copy(next_state_normalized)
        else:
            next_state =next_state_normalized

      
        return (next_state,reward1,done,info)

    def stepTest(self,aVec):
        self.countSteps+=1
        self.change_env_state()

        aVecUnique = np.unique(aVec)

        stateDict = dict()
        rewardDict = dict()

        for i in range(len(aVecUnique)):
            key = str(self.actionDict[aVecUnique[i]])+'+'+str(self.otherActionDict[self.otherActionIndex])
            next_state_full = random.choice(self.dict[key])

            next_state = self.preProcess(np.asarray(next_state_full[0:self.dimState]))
            reward = self.computeReward(next_state_full[-1])

            stateDict[aVecUnique[i]] = next_state
            rewardDict[aVecUnique[i]] = reward

        stateOut = []
        rewardOut = []

      
        if self.historyFlag:    
            for i in range(len(aVec)):
                next_state = np.concatenate([stateDict[aVec[i]],self.historyTestData[i]],axis=0)
                self.historyTestData[i] = np.roll(self.historyTestData[i],self.dimState)
                self.historyTestData[i][0:self.dimState] = np.copy(stateDict[aVec[i]])
                stateOut.append(next_state)
                rewardOut.append(rewardDict[aVec[i]])
        else:
            for i in range(len(aVec)):
                stateOut.append(stateDict[aVec[i]])
                rewardOut.append(rewardDict[aVec[i]])

        done = False

        if (self.countSteps >= self.countStepsMax):
            done = True
            #self.countIteration = 0
            #self.countIterationflag = False
            self.countSteps = 0

        info = 0

        return stateOut,rewardOut,done,info 
        
    def change_env_state(self):
        if self.transitionModel == "Markovian":
            p = np.random.uniform(0,1,1)
            if p < self.ps+0.01:
                self.otherActionIndex = not(self.otherActionIndex)
        elif self.transitionModel == "NonMarkovian":
            p = np.random.uniform(0,1,1)
            if self.incrementFlag:
                if p < self.ps+0.01:
                    if self.otherActionIndex == len(list(self.otherActionDict.keys()))-1:
                        self.otherActionIndex -= 1
                        self.incrementFlag = False 
                    else:
                        self.otherActionIndex += 1
            else:
                if p < self.ps+0.01:
                    if self.otherActionIndex == 0:
                        self.otherActionIndex += 1
                        self.incrementFlag = True
                    else:
                        self.otherActionIndex -= 1
                    

if __name__ =="__main__":

    n = 5
    ps = 1
    transitionModel = "NonMarkovian"
    history = 3
    env = envBuilder(n,ps,transitionModel,history)

    print('Reset = ', env.reset())

    print('Initial choice of others = ',env.otherActionIndex)
    for i in range(100):
        act = random.choice([0,1,2,3,4])
        s,_,_,_ = env.stepTest([act,0,act])
        print('state = ',s)
        print(env.historyTestData)
        # print("Other Node CW = ",env.otherActionIndex)

    # act = [0,1,1,2,4,4,4,2,2,1,0]
    # st,rew,done,info = env.stepTest(act)
    # print('Input Action = ',act)
    # print('State Output = ',st)
    # print('Output Reward = ',rew)  
        
    
