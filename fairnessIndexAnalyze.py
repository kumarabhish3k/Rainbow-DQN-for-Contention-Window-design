from gen_dataset import *
import numpy as np 
import pdb
import matplotlib

import matplotlib.pyplot as plt

def computeFairnessIndex(data,n):
	phiNode0 = 1/n
	phiOtherNodes = 1-phiNode0
	fIndex = (data[0]/phiNode0)/(data[1]/phiOtherNodes)
	return fIndex

def fairPolicy(fIndex,C,currCWMin):
	nextCWMin = currCWMin
	if fIndex > C:
		nextCWMin = 2*currCWMin
	elif fIndex < (1/C):
		nextCWMin = int(currCWMin/2)
	return nextCWMin

n = 10
dataDict = genDataset(n)

otherCWList = [32,128]
node0CWList = [32,64,128]

fIndexList = dict()
numSamples = 500
for i in range(len(node0CWList)):
	for j in range(len(otherCWList)):
		node0CW = node0CWList[i]
		otherCW = otherCWList[j]
		key = str(node0CW)+'+'+str(otherCW)

		data = dataDict[key]
		fIndex = []

		for k in range(numSamples):
			fIndex.append(computeFairnessIndex(data[k],n))

		fIndexList[key] =fIndex





plotData = []
plotData.append(fIndexList['32+32'])
plotData.append(fIndexList['64+32'])
plotData.append(fIndexList['128+32'])
labels = ['32+32','64+32','128+32']

fig, ax = plt.subplots()
ax.boxplot(plotData,showfliers=False,showmeans=True)
ax.set_xticklabels(labels,fontsize=10)
ax.set_ylabel('Fairness Index',fontsize = 10)
# ax.set_ylim([0,1])
ax.set_title('N = '+str(n)+', Others = 32',fontsize = 16)
ax.grid()
plt.tight_layout()
plt.savefig('./fairnessIndex/base32-'+str(n)+'Node.png')


plotData = []
plotData.append(fIndexList['32+128'])
plotData.append(fIndexList['64+128'])
plotData.append(fIndexList['128+128'])
labels = ['32+128','64+128','128+128']

fig, ax = plt.subplots()
ax.boxplot(plotData,showfliers=False,showmeans=True)
ax.set_xticklabels(labels,fontsize=10)
ax.set_ylabel('Fairness Index',fontsize = 10)
# ax.set_ylim([0,1])
ax.set_title('N = '+str(n)+', Others = 128',fontsize = 16)
ax.grid()
plt.tight_layout()
plt.savefig('./fairnessIndex/base128-'+str(n)+'Node.png')


# pdb.set_trace() 
