#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:26:24 2017

@author: repete
"""



import os
import cv2
import glob
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import detectVehicles as dv
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from skimage.feature import hog
from scipy.ndimage.measurements import label  
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split


dirData = '/home/repete/Code/SelfDriving/P5_VehicleDetection/Data'
gateVideo = False





###############################################################################
############## TRAIN CLASSIFIER ###############################################

classLabels = ['vehicles','non-vehicles'] # 
dataPerLabel = [[] for _ in classLabels]
dataPerLabel[0] = ['GTI_Far','GTI_Left','GTI_MiddleClose','GTI_Right','KITTI_extracted']
dataPerLabel[1] = ['Extras','GTI']
dataImageDirs = [[] for _ in classLabels]

for iClass, oClass in enumerate(classLabels): 
    for oIncludeData in dataPerLabel[iClass] :
        dataImageDirs[iClass] = sorted(dataImageDirs[iClass]+ glob.glob(os.path.join(dirData,oClass,oIncludeData,'*.png')))
        

# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
#sample_size = 500
#cars = dataImageDirs[0][0:sample_size]
#notcars = dataImageDirs[1][0:sample_size]


cars = dataImageDirs[0]
notcars = dataImageDirs[1]

colorspace = 'YCrCb' #'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"#"ALL" # Can be 0, 1, 2, or "ALL"
nColorBins = 32
spaFeatures = 32

# This haven't been tested.
includeFeatures = '111' # idx 0: hog, idx 1: color, idx 2: Spatial



dirFeatureFile = 'colorspace' + colorspace + '_orient' + str(orient) + '_pix_per_cell' + str(pix_per_cell) + '_cell_per_block' + str(cell_per_block) + '_hog_channel' + str(hog_channel) + '_nColorBins' + str(nColorBins) + '_spaFeatures' + str(spaFeatures) + '_includeFeatures' + includeFeatures + '.npy'
subDir = 'Features'
dirFeatures = os.path.join(dirData,subDir)
if not os.path.isdir(dirFeatures) : 
    os.mkdir(dirFeatures)
fullFile = os.path.join(dirFeatures,dirFeatureFile)


if not os.path.isfile(fullFile):
    t=time.time()
    
    car_features,featureShapesC = dv.extract_features(cars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,nbins=nColorBins,spa_features=spaFeatures,include_features=includeFeatures)
    notcar_features,featureShapesNC = dv.extract_features(notcars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,nbins=nColorBins,spa_features=spaFeatures,include_features=includeFeatures)
    
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    
    np.save(fullFile,{'X_train' : X_train, 'X_test' : X_test, 'y_train' : y_train, 'y_test' : y_test,'X_scaler':X_scaler,'featureShapes' : featureShapesC})
    featureShapes = featureShapesC
    print(round(time.time()-t, 2), 'Seconds to extract features...')
else : 
    t=time.time()
    features = np.load(fullFile).item()
    X_train = features['X_train']
    y_train = features['y_train']
    X_test = features['X_test']
    y_test = features['y_test']
    X_scaler = features['X_scaler']
    featureShapes = features['featureShapes']
    print(round(time.time()-t, 2), 'Seconds to load features...')

for oFeatureShape in featureShapes : 
    print("FeatureShapes: ", oFeatureShape)
    
classifierTypes = ['LinearSVC','Adaboost']
classifierType = classifierTypes[0]

subDirClassifiers = 'Classifiers'
dirClassifiers = os.path.join(dirData,subDirClassifiers)
dirClassifierFile = 'Classifier' + classifierType + '__' +  dirFeatureFile
if not os.path.isdir(dirClassifiers) : 
    os.mkdir(dirClassifiers)
fullFileClassifier = os.path.join(dirClassifiers,dirClassifierFile)




# Check the training time for the SVC
# Use a linear SVC 
# Split up data into randomized training and test sets

print('Feature vector length:', len(X_train[0]))    
if not os.path.isfile(fullFileClassifier) : 
    if classifierType == 'LinearSVC' : 
        clf = LinearSVC()
    if classifierType == 'Adaboost' : 
        clf = AdaBoostClassifier()
    
    #print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
    
    t=time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    np.save(fullFileClassifier,{'clf':clf})
else : 
    t=time.time() 
    clf = np.load(fullFileClassifier).item()['clf']


    print(round(time.time()-t, 4), 'Seconds to load classifier...')


# Check the score of the SVC
print('Test Accuracy of ' + classifierType +' = ', round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
testPredictions =  clf.predict(X_test[0:n_predict])
#print('Predicts: ', testPredictions)
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels')



###############################################################################
############## RUN CLASSIFIER #################################################

print("X_scaler",X_scaler,"orient",orient,"pix_per_cell",pix_per_cell,"cell_per_block",cell_per_block,"spatial_size",spaFeatures,"nColorBins",nColorBins)
img = mpimg.imread('test_images/test1.jpg')




  
ystart = 400
ystop = 656
scale = 1.5
threshold = 1

    
imgAllBB,heatmap = dv.find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, (spaFeatures,spaFeatures), nColorBins)

#threshold = 1
bwCar = dv.apply_threshold(heatmap.copy(), threshold)
labels = label(bwCar)

imgBB = dv.draw_labeled_bboxes(img,labels)


#plt.figure()
#plt.imshow(mpimg.imread(cars[0]))
#plt.savefig('output_images/CarExample1.png', bbox_inches='tight')
#
#plt.figure()
#plt.imshow(mpimg.imread(cars[100]))
#plt.savefig('output_images/CarExample2.png', bbox_inches='tight')
#
#plt.figure()
#plt.imshow(mpimg.imread(notcars[0]))
#plt.savefig('output_images/NonCarExample1.png', bbox_inches='tight')
#
#plt.figure()
#plt.imshow(mpimg.imread(notcars[100]))
#plt.savefig('output_images/NonCarExample2.png', bbox_inches='tight')


#plt.figure()
#plt.imshow(imgAllBB)
#plt.savefig('output_images/AllBoundingBoxes.png', bbox_inches='tight')
#
#plt.figure()
#plt.imshow(heatmap)
#plt.savefig('output_images/Heatmap.png', bbox_inches='tight')
#
#plt.figure()
#plt.imshow(bwCar)
#plt.savefig('output_images/ThresholdedHeatmat.png', bbox_inches='tight')
#
#print(labels[1], 'cars found')
#plt.figure()
#plt.imshow(labels[0],cmap='gray')
#plt.savefig('output_images/LabelImage.png', bbox_inches='tight')
#
#plt.figure()
#plt.imshow(imgBB,cmap='gray')
#plt.savefig('output_images/DetectedCars.png', bbox_inches='tight')





# For processing video.
if gateVideo : 
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cFrames = 1
    # Directory of input video.
    dirInputVideo = "project_video"
    #dirInputVideo = "test_video"
    clip1 = VideoFileClip(dirInputVideo+".mp4")
    
    print("Loading video...")
    images = [rgb for rgb  in clip1.iter_frames()]
    print("Done loading video")    
    
    # Keeps info of previous frames.
    keepInfo = 0.8
    scales = [0.50, 1.5]
    #scales = [1.5]
    threshold = 1
    heatRecurrent = np.zeros((images[0].shape[0],images[0].shape[1]))
    
    imagesResult = []
    #images = [images[0]]
    #images = images[0:200]
    # Step through all images in the video
    for idxImage,img in enumerate(images): 
        
        #img = mpimg.imread('test_images/test1.jpg')
        print("Current image: ", idxImage+1,"/",len(images))
        heatmap = np.zeros_like(heatRecurrent)
        for oScale in scales : 
            _,heatmapTmp = dv.find_cars(img, ystart, ystop, oScale, clf, X_scaler, orient, pix_per_cell, cell_per_block, (spaFeatures,spaFeatures), nColorBins)
            heatmap = heatmap+heatmapTmp
        #_,heatmap1 = find_cars(img, ystart, ystop, scales[0], clf, X_scaler, orient, pix_per_cell, cell_per_block, (spaFeatures,spaFeatures), nColorBins)
#        _,heatmap2 = find_cars(img, ystart, ystop, scales[1], clf, X_scaler, orient, pix_per_cell, cell_per_block, (spaFeatures,spaFeatures), nColorBins)
#        heatmap = heatmap1+heatmap2
#        if idxImage == 0 :
#            heatRecurrent = heatmap
#        else: 
        heatRecurrent = heatRecurrent*keepInfo+heatmap
        
        bwCar = dv.apply_threshold(heatRecurrent, threshold)
        labels = label(bwCar)
        
        imgBB = dv.draw_labeled_bboxes(img,labels)
        imagesResult.append(imgBB)
        cv2.imshow('image',cv2.cvtColor(imgBB,cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    
    
     # Create video from images. 
    new_clip = ImageSequenceClip(imagesResult, fps=clip1.fps)
    new_clip.write_videofile(dirInputVideo + "_Processed.mp4") 