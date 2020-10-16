import numpy as np 
import random
import pickle
import pdb
from math import *
import matplotlib.pyplot as plt 
from pathlib import Path

def plotReward(n,cw2,cw1List,sampleSize,baseFolder):
    # Fairness Box Plot 
    omega = (1.0/n)*np.ones((sampleSize))
    data_dict = dict()
    cw_dict=dict()
    dataFolder = str(n)+'Node'
    k=0
    for i in range(len(cw1List)):
            # print(cw1List[i],cw2List[j])
            node1TxPackets = np.loadtxt('./Dataset/'+dataFolder+'/node1TxPackets'+'+'+str(cw1List[i])+'+'+str(cw2)+'.txt')
                
            node1RxPackets = np.loadtxt('./Dataset/'+dataFolder+'/node1RxPackets'+'+'+str(cw1List[i])+'+'+str(cw2)+'.txt')
                
            otherRxPackets = np.loadtxt('./Dataset/'+dataFolder+'/otherRxPackets'+'+'+str(cw1List[i])+'+'+str(cw2)+'.txt')
            

            # Compute Reward
            totalPackets = node1RxPackets+otherRxPackets
            rho = np.divide(node1RxPackets,totalPackets)
            reward = np.abs(rho-omega)

            key = str(cw1List[i])+'+'+str(cw2)
            data_dict[k] = (1-reward)
            cw_dict[k] = key
            k+=1
    # pdb.set_trace()
    data = list(data_dict.values())
    data = np.asarray(data)
    data = data.T

    data_avg = np.mean(data,0)
    # pdb.set_trace()

    labels = list(cw_dict.values())

    fig, ax = plt.subplots()
    ax.boxplot(data,showfliers=False,showmeans=True)
    ax.set_xticklabels(labels,fontsize=5)
    ax.set_ylabel('Utility(Fairness)',fontsize = 10)
    ax.set_ylim([0,1])
    ax.set_title('N = '+str(n)+', cw = '+str(cw2),fontsize = 16)
    ax.grid()
    plt.tight_layout()
    # ax.plot(data_avg)
    plt.savefig(baseFolder+'plot'+str(n)+'NodeOther'+str(cw2)+'.pdf')

if __name__ == '__main__':
    nList = [5,10,20,40]
    cw2List = [32,64,128,256,512]
    sampleSize = 500
    cwLow = 32
    cwHigh = 128
    cwDiff = 8

    actionDim = 9;

    cw1List = [32,48,64,96,128,192,256,384,512];
    
    baseFolder = './Dataset/plots/'
    Path(baseFolder).mkdir(parents=True, exist_ok=True)

    for n in nList:
        for cw2 in cw2List:
            plotReward(n,cw2,cw1List,sampleSize,baseFolder)




