from simulation import *

def envBuilder(n,ps,transitionModel,history):
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

	env = commEnv(n,ps,transitionModel,actionDict,otherActionDict,epLen,history)
	return env