from gen_dataset import *
import numpy as np 
import pdb
import matplotlib
from envBuilder import *
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def computeFairnessIndex(data,n):
	phiNode0 = 1/n
	phiOtherNodes = 1-phiNode0
	fIndex = (data[0]/phiNode0)/(data[1]/phiOtherNodes)
	return fIndex

def fairPolicy(fIndex,C,currCWMin,node0CWList):
	nextCWMin = currCWMin
	if fIndex > C:
		nextCWMin = np.min([2*currCWMin,node0CWList[-1]])
	elif fIndex < (1/C):
		nextCWMin = np.max([int(currCWMin/2),node0CWList[0]])
	return nextCWMin





parser = argparse.ArgumentParser()
parser.add_argument("n", type=np.int32, help="Number of nodes")
parser.add_argument("ps", type=np.float32, help="Transition Probability")
parser.add_argument("transitionModel", type=str, help="Transition Model")

args = parser.parse_args()

n = args.n
ps = args.ps
transitionModel = args.transitionModel
history = 0
env = envBuilder(n,ps,transitionModel,history)
env.reset()

if transitionModel=="Markovian":
	node0CWList = [32,64,128]
	otherCWList = [32,128]
elif transitionModel=="NonMarkovian":
	node0CWList = [32,64,128,256,512]
	otherCWList = [32,64,128,256,512]



numEpisodes = 500
lenEpisode = 50

gamma = 1
rewardList = []
C = 1.5
updateSteps = 3

for i in range(numEpisodes):
	node0CW = random.choice(node0CWList)
	otherCW = env.otherActionDict[env.otherActionIndex]
	count = 0
	done = False
	transitionFlag = False
	countTransition = 0
	reward = 0
	while not(done):
		
		if transitionFlag:
			env.change_env_state()
			otherCW = env.otherActionDict[env.otherActionIndex]
			transitionFlag = False
		# print('Node 0 CW = ',node0CW,'Other CW = ',otherCW)
		key = str(node0CW)+'+'+str(otherCW)
		data = random.choice(env.dict[key])
		
		r = 1-data[-1]
		reward+=(gamma**count)*r

		# if (r<0.5):
			# print('Hello')

		# print( 'Other CW = ',otherCW,'Node 0 CW = ',node0CW)
		# print('reward = ',reward)
		fIndex = computeFairnessIndex(data,n)
		newCW = fairPolicy(fIndex,C,node0CW,node0CWList)
		node0CW = newCW
		count+=1
		countTransition+=1
		if countTransition==updateSteps:
			transitionFlag=True
			countTransition = 0


		if count==lenEpisode:
			done = True

	rewardList.append(reward/lenEpisode)
baseFolder = './fairnessIndex/' 
Path(baseFolder).mkdir(parents=True, exist_ok=True)
path = baseFolder+str(n)+'Node'+transitionModel+'ps'+str(int(100*ps))+'.txt'
np.savetxt(path,rewardList,delimiter=',')
print('reward avg = ',np.mean(rewardList))