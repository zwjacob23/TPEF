from tqdm import tqdm
import time
import data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
from shared import dataset_tools
import randomcut

def transformPressure(pressure):
    pressure = np.clip((pressure.astype(np.float32) - 500) / (650 - 500), 0.0, 1.0)
    return pressure


def sigmoid(x):
     return 1/(1+np.exp(-x))

meta = dataset_tools.loadMetadata('metadata.mat')
pressure = transformPressure(meta['pressure'])
mask = np.logical_and(meta['hasValidLabel']==1,meta['isBalanced']==1)
indices = np.argwhere(mask)

objectId_ori = list(meta['objectId'][indices])
recordingId_ori = list(meta['recordingId'][indices])
splitId_ori = list(meta['splitId'][indices])
pressure_ori = pressure[indices]
pressure_ori = pressure_ori.reshape(pressure_ori.shape[0],pressure_ori.shape[-1],pressure_ori.shape[-1])

mask_train = np.logical_and(mask,meta['splitId']==0)
indices_train = np.argwhere(mask_train)

pressure_trained = pressure[indices_train]
pressure_trained = np.squeeze(pressure_trained)
objectId_trained = list(meta['objectId'][indices_train])
recordingId_trained = list(meta['recordingId'][indices_train])
splitId_trained = list(meta['splitId'][indices_train])

pressure_filled = np.zeros((pressure_trained.shape[0],pressure_trained.shape[1],pressure_trained.shape[2]))

len = pressure_filled.shape[0]

for frame in tqdm(range(len)):
    single = pressure_trained[frame].reshape(32, 32)
    match = (single < 0.07)
    for i in range(32):
        for j in range(32):
            if match[i][j] == 0:
                continue
            else:
                index_row = match[i, :] == 0
                index_column = match[:, j] == 0
                data_row = single[i, :]
                data_column = single[:, j]
                ans = (np.sum(data_column[index_column]) + np.sum(data_row[index_row])) / (
                            (np.argwhere(index_column).shape[0]) + (np.argwhere(index_row).shape[0]))
                single[i][j] = sigmoid(ans)
                match[i][j] = 0

    single = single.reshape(1, 32, 32)
    pressure_filled[frame] = single

pressure_randomcut = randomcut.randomcutt(pressure_filled,np.array(recordingId_trained))
pressure = np.concatenate((pressure_ori,pressure_randomcut),axis=0)*255.0
objectId = np.array(objectId_ori+objectId_trained)
recordingId = np.array(recordingId_ori+recordingId_trained)
splitId = np.array(splitId_ori+splitId_trained)
sio.savemat('metadata_tacmix_c.mat',{'recordingId':recordingId,'objectId':objectId,'pressure':pressure,'splitId':splitId})
