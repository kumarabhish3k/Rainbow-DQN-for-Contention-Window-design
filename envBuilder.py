from simulation import *

def envBuilder(n,ps,transitionModel,history):
    '''
    n: wireless network size; {5,10,20}

    ps: transition probability of stochastic process; {0.75,0.9,1}

    transitionModel: stochastic model followed by other nodes; {'Markovian', 'NonMarkovian'} (Here 'NonMarkovian' corresponds to the 'complex' process in the paper.)

    history: number of previous time steps from which observations are used as input; {0,1,2,3}

    '''
    epLen = 50
    actionDict = dict()
    otherActionDict = dict()

    if transitionModel == "Markovian":
        actions = [32,48,64,96,128]
        # num_action = len(actions)
        for i in range(len(actions)):
            actionDict[i]=actions[i]

        otherActions = [32,128]
        for i in range(len(otherActions)):
            otherActionDict[i]=otherActions[i]

    elif transitionModel == "NonMarkovian":
        actions = [32,48,64,96,128,192,256,384,512]
        # num_action = len(actions)
        for i in range(len(actions)):
            actionDict[i]=actions[i]

        otherActions = [32,64,128,256,512]
        for i in range(len(otherActions)):
            otherActionDict[i]=otherActions[i]

    env = commEnv(n, ps, transitionModel, actionDict, otherActionDict, epLen, history)
    return env