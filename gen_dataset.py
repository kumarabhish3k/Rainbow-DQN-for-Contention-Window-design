import numpy as np 
import random
import pickle
import pdb


def genDataset(n):
    '''
    output (data_dict) a dictionary
    '''
    n = n

    # For computing reward
    sampleSize = 500
    omega = (1.0/n)*np.ones((sampleSize))

    cw1List = [32,48,64,96,128,192,256,384,512]
    cw2List = [32,64,128,256,512]
    actionDim = 9

    # Fairness Box Plot 
    data_dict = dict()

    dataFolder = str(n)+'Node'
    for i in range(actionDim):
        for j in range(len(cw2List)):
            
            # print(cw1List[i],cw2List[j])
            node1TxPackets = np.loadtxt('./Dataset/'+dataFolder+'/node1TxPackets'+'+'+str(cw1List[i])+'+'+str(cw2List[j])+'.txt')
            
            node1RxPackets = np.loadtxt('./Dataset/'+dataFolder+'/node1RxPackets'+'+'+str(cw1List[i])+'+'+str(cw2List[j])+'.txt')
            
            otherRxPackets = np.loadtxt('./Dataset/'+dataFolder+'/otherRxPackets'+'+'+str(cw1List[i])+'+'+str(cw2List[j])+'.txt')

            # Compute Reward
            totalPackets = node1RxPackets+otherRxPackets
            rho = np.divide(node1RxPackets,totalPackets)
            reward = np.abs(rho-omega)
            
            dataset= []
            for k in range(sampleSize):
                dataTemp = [node1RxPackets[k],otherRxPackets[k],cw1List[i],reward[k]]
                dataset.append(dataTemp)

            key = str(cw1List[i])+'+'+str(cw2List[j])
            data_dict[key] = dataset

    return data_dict

