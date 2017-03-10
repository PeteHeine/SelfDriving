#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:02:50 2017

@author: repete
"""
import numpy as np 
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc
# SMALL ADJUSTMENS TO TRAINING:
# 1) Remove header names from udacity data.
# 2) In data_recover_left_mikkel_1: Remove data recorded at center_2017_01_17_10_10_00_415.jpg

dirData = "/home/repete/DataFolder/SelfDriving/MultipleSets"
dirDataOut = "/home/repete/DataFolder/SelfDriving/CollectedSets"

selectedNames = sorted(['peter','mikkel','udacity']) # Data to be included.
smoothSteering = sorted(['peter','mikkel','udacity'])
imageResizing = 4
nSize = 5

dirFolders = glob.glob(os.path.join(dirData,"data*"))
selectedFolders = dirFolders
#selectedFolders = dirFolders[(0,1)]
#selectedFolders = [dirFolders[idx] for idx in (1,2)]
#selectedFolders = [dirFolders[0]]

newDf = pd.DataFrame(columns=['center','left','right','steering','throttle','break','speed'])

outputName = "Resizing" + str(imageResizing) + "_AddedData" + ''.join(selectedNames) + '_SmoothSteeringAmount' + str(nSize) + "To" +''.join(smoothSteering)
dirDataOutFull = os.path.join(dirDataOut,outputName)

### Creates new folder 
if(not os.path.isdir(dirDataOutFull)):
    os.mkdir(dirDataOutFull)
    os.mkdir(os.path.join(dirDataOutFull,"IMG"))
cIdx = 0
#newDf = newDf.append(pd.DataFrame([1,2,3,4,5,6,7],columns=['center','left','right','steering','throttle','break','speed']))
for cDir, iDir in enumerate(selectedFolders):
    lap = iDir.split("_")[-1]
    name = iDir.split("_")[-2]
    lapType = iDir.split("_")[-3]
    # Use only data specified in selectedNames-variable.
    if name in selectedNames:
        # Data from self-made and udacity data is not read in the same way.
        data = pd.read_csv(os.path.join(iDir,"driving_log.csv"),names=['center','left','right','steering','throttle','break','speed'])
        addStr = ''
        
        ### Add smoothing to steering angle
        
        # Smoothing is only performed on peter and mikkel (meaning, not on the udacity data)
        if name in smoothSteering:
            win = (np.ones(nSize)/nSize)
            steeringFiltered = signal.convolve(data['steering'],win,mode='same')
            addStr = addStr + "AvgFilter, "
            data['steering'] = steeringFiltered

        ### Removing index in recovery mode.
        #print(lapType)
        if lapType == 'left': # in recovery mode
            data = data[data['steering']<0] # Use only image with negative angles. 
            addStr = addStr + "leftRecover, "
        
        if lapType == 'right': # in recovery mode
            data = data[data['steering']>0] # Use only image with positive angles. 
            addStr = addStr + "rightRecover, "
        
        print('Folder: (',str(cDir+1),'/',len(selectedFolders),') dir: ',   iDir, ', Tricks: ', addStr)
        ### Iterate through all images
        for iImg in data.transpose():
            newImgName = dict({})
            for iiImg in ['center','left','right']:
                imgName = data[iiImg][iImg].split("/")[-1] # Image name from csv
                img = misc.imread(os.path.join(iDir,"IMG",imgName))
                imgResized = misc.imresize(img,(int(img.shape[0]/imageResizing),int(img.shape[1]/imageResizing)))
                newImgName[iiImg] = os.path.join(dirDataOutFull,"IMG",imgName)
                misc.imsave(newImgName[iiImg] ,imgResized)
                
            newDf.loc[cIdx] = [newImgName['center'],newImgName['left'],newImgName['right'],data['steering'][iImg],data['throttle'][iImg],data['break'][iImg],data['speed'][iImg]]
            cIdx = cIdx+1
            if(cIdx%25 == 0):
                print("NumberOfLoadImages:", str(cIdx), 'Folder: (',str(cDir+1),'/',len(selectedFolders),')',"(", str(iImg+1),"/",str(data.shape[0]),")"  )
        del data
### Store and visualize data. 
newDf.to_csv(os.path.join(dirDataOutFull,"driving_log.csv"))