from simulation import *
import numpy as np
import argparse
import pickle
from gen_dataset import *
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from envBuilder import *
from pathlib import Path


def getDatasetRF(env,N,num_action):
    state = env.reset()
    X=[]
    y=[]

    for i in range(num_action):
        print('Training i =  ',i+1)
        for j in range(N):
            action = i
            # print('action',action)
            next_state_list = env.step(action)
            X.append(state)
            state = next_state_list[0]
            y.append(int(env.otherActionIndex))
    return X,y

parser = argparse.ArgumentParser()
parser.add_argument("n", type=np.int32, help="Number of nodes")
parser.add_argument("ps", type=np.float32, help="Transition Probability")
parser.add_argument("transitionModel", type=str, help="Transition Model")
parser.add_argument("history", type=np.int32, help="History Level")
args = parser.parse_args()

# Create environment
n = args.n
ps = args.ps
print('ps = ',ps,'100ps = ',int(100*ps))
transitionModel = args.transitionModel
history = args.history

env = envBuilder(n,ps,transitionModel,history)
num_action = len(list(env.actionDict.keys()))

numTrain = 300
xTrain,yTrain = getDatasetRF(env,numTrain,num_action)

numTest = 500
xTest,yTest = getDatasetRF(env,numTest,num_action)

baseFolder ='./modelRF/'+str(n)+'Node/'+transitionModel+'/'+str(history)+'history/ps'+str(int(100*ps))+"/" 
Path(baseFolder).mkdir(parents=True, exist_ok=True)

np.savetxt(baseFolder+'xtrain.txt',xTrain,delimiter = ',')
np.savetxt(baseFolder+'ytrain.txt',yTrain,delimiter = ',')


# Training RF
clf = RandomForestClassifier(n_estimators=20 ,max_depth=15, random_state=0)
clf.fit(np.asarray(xTrain), np.asarray(yTrain))
ypred = clf.predict(xTrain)
print('Training Accuracy = ',100*(ypred.shape[0]-np.sum(np.abs(ypred-yTrain)))/ypred.shape[0])


dump(clf, baseFolder+'rf.joblib') 
# clf2 = load('rf.joblib') 
ypred2 = clf.predict(xTest)
print('Test Accuracy = ',100*(ypred2.shape[0]-np.sum(np.abs(ypred2-yTest)))/ypred2.shape[0])


