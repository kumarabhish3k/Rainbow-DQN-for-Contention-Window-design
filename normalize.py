import random
import numpy as np
from gen_dataset import *
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("n", type=np.int32, help="Number of nodes")
args = parser.parse_args()
    # Create environment
n = args.n
dict1 = genDataset(n)

# For computing reward
sampleSize = 500

cw1List = [32,48,64,96,128,192,256,384,512]
cw2List = [32,64,128,256,512]
actionDim = 9

new_data = []

for i in range(actionDim):
	for j in range(len(cw2List)):
		key = str(cw1List[i])+'+'+str(cw2List[j])
		data = np.asarray(dict1[key])

		for k in range(sampleSize):
			new_data.append(data[k,:-1])

new_data = np.asarray(new_data)

data_mean = np.mean(new_data,0)
data_std = np.std(new_data,0)


baseFolder = './Dataset/dataStats/'+str(n)+'Node/'

Path(baseFolder).mkdir(parents=True, exist_ok=True)

np.savetxt(baseFolder+'data_mean.txt',data_mean,delimiter =  ',')
np.savetxt(baseFolder+'data_std.txt',data_std,delimiter =  ',')