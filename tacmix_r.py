from tqdm import tqdm
import time
import dataset_tools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import randomcut
import os
import scipy.io as sio
def transformPressure(pressure):
    pressure = np.clip((pressure.astype(np.float32) - 500) / (650 - 500), 0.0, 1.0)
    return pressure

step = [(1,-1),(1,0),(1,1),(1,0),(-1,1),(-1,0),(-1,-1),(-1,0)]#边长为3

step_5 = [
    (2, -2), (2, -1), (2, 0), (2, 1), (2, 2),
    (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
    (0, -2), (0, -1), (0, 1), (0, 2),
    (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
    (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2)
]
step_7 = [
    (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3),
    (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3),
    (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3),
    (0, -3), (0, -2), (0, -1),  (0, 1), (0, 2), (0, 3),
    (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3),
    (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3),
    (-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3)
]
step_9 = [
    (4, -4), (4, -3), (4, -2), (4, -1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
    (3, -4), (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
    (2, -4), (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    (1, -4), (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
    (0, -4), (0, -3), (0, -2), (0, -1),  (0, 1), (0, 2), (0, 3), (0, 4),
    (-1, -4), (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4),
    (-2, -4), (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3), (-2, 4),
    (-3, -4), (-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3), (-3, 4),
    (-4, -4), (-4, -3), (-4, -2), (-4, -1), (-4, 0), (-4, 1), (-4, 2), (-4, 3), (-4, 4)
]
step_12 = [
    (6, -6), (6, -5), (6, -4), (6, -3), (6, -2), (6, -1), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
    (5, -6), (5, -5), (5, -4), (5, -3), (5, -2), (5, -1), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
    (4, -6), (4, -5), (4, -4), (4, -3), (4, -2), (4, -1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (3, -6), (3, -5), (3, -4), (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
    (2, -6), (2, -5), (2, -4), (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (1, -6), (1, -5), (1, -4), (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (0, -6), (0, -5), (0, -4), (0, -3), (0, -2), (0, -1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
    (-1, -6), (-1, -5), (-1, -4), (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, 5), (-1, 6),
    (-2, -6), (-2, -5), (-2, -4), (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3), (-2, 4), (-2, 5), (-2, 6),
    (-3, -6), (-3, -5), (-3, -4), (-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3), (-3, 4), (-3, 5), (-3, 6),
    (-4, -6), (-4, -5), (-4, -4), (-4, -3), (-4, -2), (-4, -1), (-4, 0), (-4, 1), (-4, 2), (-4, 3), (-4, 4), (-4, 5), (-4, 6),
    (-5, -6), (-5, -5), (-5, -4), (-5, -3), (-5, -2), (-5, -1), (-5, 0), (-5, 1), (-5, 2), (-5, 3), (-5, 4), (-5, 5), (-5, 6),
    (-6, -6), (-6, -5), (-6, -4), (-6, -3), (-6, -2), (-6, -1), (-6, 0), (-6, 1), (-6, 2), (-6, 3), (-6, 4), (-6, 5), (-6, 6)
]

meta = dataset_tools.loadMetadata('./metadata.mat')

pressure = transformPressure(meta['pressure'])

mask = np.logical_and(meta['hasValidLabel']==1,meta['isBalanced']==1)
indices = np.argwhere(mask)

objectId_ori = list(meta['objectId'][indices])
recordingId_ori = list(meta['recordingId'][indices])
splitId_ori = list(meta['splitId'][indices])

pressure_ori = pressure
pressure_ori = pressure_ori.reshape(pressure_ori.shape[0],pressure_ori.shape[-1],pressure_ori.shape[-1])

mask_train = np.logical_and(mask,meta['splitId']==0)

indices_train = np.argwhere(mask_train)

pressure_trained = pressure[indices_train]
objectId_trained = list(meta['objectId'][indices_train])
recordingId_trained = list(meta['recordingId'][indices_train])
splitId_trained = list(meta['splitId'][indices_train])

pressure_filled = np.zeros((pressure_trained.shape[0],pressure_trained.shape[2],pressure_trained.shape[3]))

len = pressure_filled.shape[0]

island_p = [(18,24,25,14),(13,13,14,2),(9,13,14,1),(3,13,14,3),(18,29,3,14)]

for frame in tqdm(range(len)):
    single = pressure_trained[frame].reshape(32,32)
    match = (single < 0.07).reshape(32, 32)
    kk=1
    for start_x,start_y,len_x,len_y in island_p:

        for i in range(len_x):
            for j in range(len_y):
                x = (start_x+j,start_y-i) if kk!=5 else (start_x+j,start_y+i)
                sum = []
                for dx,dy in step_5:
                    new_x = x[0]+dx
                    new_y = x[1]+dy
                    if(0<(new_x)<32 and 0<(new_y)<32):
                        if match[new_x][new_y] == 1:
                            continue
                        else:
                            sum.append(single[new_x][new_y])
                    else:
                        continue
                single[x[0]][x[1]] = np.average(sum)
                match[x[0]][x[1]] = 0
        kk = kk+1
    single = single.reshape(1,32,32)
    pressure_filled[frame] = single

pressure_randomcut = randomcut.randomcutt(pressure_filled,np.array(recordingId_trained.copy()))

pressure = np.concatenate((pressure_ori,pressure_randomcut),axis=0)*255.0
objectId = np.array(np.uint8(objectId_ori + objectId_trained))
recordingId = np.array(np.uint8(recordingId_ori + recordingId_trained))
splitId = np.array(np.uint8(splitId_ori + splitId_trained))

for i in range(objectId.shape[0]):
    objectId[i] = np.uint8(objectId[i])
    recordingId[i] = np.uint8(recordingId[i])
    splitId[i] = np.uint8(splitId[i])

sio.savemat('metadata_tacmix_r.mat',{'recordingId':recordingId,'objectId':objectId,'pressure':pressure,'splitId':splitId})

