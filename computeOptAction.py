import numpy as np 
from gen_dataset import *
from envBuilder import *


def computeOptAction(env):
    if env.transitionModel == "Markovian":
        currentOtherIndex = env.otherActionIndex
        # This is because ps is always greater than 0.5 in our experiments
        nextPossibleOtherIndex = not(env.otherActionIndex)

    if env.transitionModel == "NonMarkovian":
        currentOtherIndex = env.otherActionIndex
        if env.incrementFlag:
            if env.otherActionIndex == len(list(env.otherActionDict.keys()))-1:
                nextPossibleOtherIndex = env.otherActionIndex-1
            else:
                nextPossibleOtherIndex = env.otherActionIndex+1
        else:
            if env.otherActionIndex == 0:
                nextPossibleOtherIndex = env.otherActionIndex+1
            else:
                nextPossibleOtherIndex = env.otherActionIndex-1

    sameStateReward = []
    changeStateReward = []

    for i in range(len(env.actionDict)):
        key = str(env.actionDict[i])+'+'+str(env.otherActionDict[currentOtherIndex])
        data = env.dict[key]
        data = np.asarray(data) 
        sameStateReward.append(1-np.mean(data[:,-1]))

        key = str(env.actionDict[i])+'+'+str(env.otherActionDict[nextPossibleOtherIndex])
        data = env.dict[key]
        data = np.asarray(data) 
        changeStateReward.append(1-np.mean(data[:,-1]))

    sameStateReward = np.asarray(sameStateReward)
    changeStateReward = np.asarray(changeStateReward)

    expectedReward = ((1-env.ps)*sameStateReward)+(env.ps*changeStateReward)
    
    optActionIndex = np.argmax(expectedReward)

    return optActionIndex


if __name__ == '__main__':
    transitionModel = "NonMarkovian"
    ps = 0.75
    n = 10
    history = 0

    env = envBuilder(n,ps,transitionModel,history)
    env.reset()
    
    optActionIndex = computeOptAction(env)
    print('Current CW = ',env.otherActionDict[env.otherActionIndex])
    print('Opt Action = ',env.actionDict[optActionIndex])
        
    for i in range(20):
        env.step(0)
        optActionIndex = computeOptAction(env)
        print('Current CW = ',env.otherActionDict[env.otherActionIndex])
        print('Opt Action = ',env.actionDict[optActionIndex])





