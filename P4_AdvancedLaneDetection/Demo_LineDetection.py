#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:49:27 2017

@author: repete
"""
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import lineDetection as ld

    
# Visualization gates. Specifies what to be visualized. 
visualizeCalibration = False
visualizeAndSelectThreshold = False
thresholdType = 3
visualizeAndSelectPerspective = True
visualizeSlidingWindow = False
visualizePolyfit = False
visualizeResult = 0 # 0,1 or 2 (subplot of the 8 test images)

# Processing gates. 
gateTestImages = False # Run test images
gateVideo = True # Run video


useColorDetection = True

# Calibration images.
dirCalibrationImages = './camera_cal'
dirsCalibrationImages = sorted(glob.glob(os.path.join(dirCalibrationImages,'*.jpg')))

# Test images
dirTestImages = './test_images'
dirsTestImages = sorted(glob.glob(os.path.join(dirTestImages,'*.jpg')))    

# Calibration file
    # To create new calibration: Delete the existing calibration file. RUN: os.remove(dirCalibrationOutput)
dirCalibrationOutput = os.path.join(dirCalibrationImages,'CalibrationData.npy')

# Make camera calibration or load a previous camera calibration file. 
if not os.path.isfile(dirCalibrationOutput) : 
    # Checkerbord grid
    checkerboardGrid = (9,6)
    # Calibration is performed in the myCameraCalibration-function.
    calibrationData = ld.myCameraCalibration(dirsCalibrationImages,checkerboardGrid)
    np.save(dirCalibrationOutput,calibrationData)
else :
    calibrationData = np.load(dirCalibrationOutput).item()

if visualizeCalibration : 
    
    # To demonstrate rectification of checkerboard image. 
    rgb = cv2.imread(dirsCalibrationImages[0])
    # Distortion correction
    camMat = calibrationData['cameraMatrix']
    distParam = calibrationData['distortionParam']
    # Undistortion is handled using opencv.
    rgbCor = cv2.undistort(rgb, camMat, distParam, None, camMat)
    
    plt.figure()
    plt.subplot(1,2,1)
    ld.imshow_bgr(plt,rgb)
    plt.subplot(1,2,2)
    ld.imshow_bgr(plt,rgbCor)
    
    plt.savefig('./examples/RectificationCheckerboard.png', bbox_inches='tight')
    
    # To demonstrate rectification of test image. 
    rgb = cv2.imread(dirsTestImages[6])
    # Distortion correction
    camMat = calibrationData['cameraMatrix']
    distParam = calibrationData['distortionParam']
    # Undistortion is handled using opencv.
    rgbCor = cv2.undistort(rgb, camMat, distParam, None, camMat)
    
    plt.figure()
    plt.subplot(1,2,1)
    ld.imshow_bgr(plt,rgb)
    plt.subplot(1,2,2)
    ld.imshow_bgr(plt,rgbCor)
    
    plt.savefig('./examples/RectificationRoadImage.png', bbox_inches='tight')
    
# Crop input image - only the bottom of the image is used. 
imgCropAbs = [0.4, 1.0, 0.0, 1.0]


## This section for selecting appropriate thresholds.
if visualizeAndSelectThreshold : 
    # First cropped images are stacked in a single image.
    for idxImage,iDirImage in enumerate(dirsTestImages): 
        print("Current image: ", idxImage+1,"/",len(dirsTestImages))
        
        # Read image
        rgb = cv2.imread(iDirImage)
        dimRgb = rgb.shape
    
        # Distortion correction
        camMat = calibrationData['cameraMatrix']
        distParam = calibrationData['distortionParam']
        # Undistortion is handled using opencv.
        rgbCor = cv2.undistort(rgb, camMat, distParam, None, camMat)    
        
        # Crop images
        imgCrop = imgCropAbs*np.array([dimRgb[0],dimRgb[0],dimRgb[1],dimRgb[1]])
        rgbCrop = rgbCor[imgCrop[0]:imgCrop[1],imgCrop[2]:imgCrop[3],:]
    
        # Visualize input image, undistored and cropped image for the first test image.
        if idxImage == 0 : 
            plt.figure()
            plt.subplot(1,3,1)
            ld.imshow_bgr(plt,rgb)
            plt.title("Distorted")
            
            plt.subplot(1,3,2)
            ld.imshow_bgr(plt,rgbCor)
            plt.title("Distortion corrected")    
            
            plt.subplot(1,3,3)
            ld.imshow_bgr(plt,rgbCrop)
            plt.title("cropped")
        
        # The cropped and undistored images are stacked to a very high image.
        if idxImage == 0 : 
            rgbStacked = rgbCrop
        else : 
            rgbStacked = np.vstack((rgbStacked,rgbCrop))
    # Show stacked images.
    plt.figure()
    ld.imshow_bgr(plt,rgbStacked)
    
    
    # Function: GUI window to select appropriate threshold values.
        # thresholdType = 0: Select threshold for sobelx and sobely
        # thresholdType = 1: Select threshold for magnitude and direction of gradient 
        # thresholdType = 2: Select threshold for l and s in hls color space.
    ld.visualizeAndSelectThreshold(rgbStacked,thresholdType)


# Read image
rgb = cv2.imread(dirsTestImages[0])
dimRgb = rgb.shape

## This section if for unwrapping images.

# Load calibration data to undistortion image.
camMat = calibrationData['cameraMatrix']
distParam = calibrationData['distortionParam']
rgbCor = cv2.undistort(rgb, camMat, distParam, None, camMat)    

# Image is cropped
imgCrop = (imgCropAbs*np.array([dimRgb[0],dimRgb[0],dimRgb[1],dimRgb[1]])).astype(np.int)
rgbCrop = rgbCor[imgCrop[0]:imgCrop[1],imgCrop[2]:imgCrop[3],:]

# The sourch and distination points are selected 
#pSrc = np.array([[200,430],[1100,430],[689,162],[592,162]],dtype=np.float32)
pSrc = np.array([[200,430],[1100,430],[715, 180],[570, 180]],dtype=np.float32)
pDist = np.array([[400,430],[900,430],[900,162],[400,162]],dtype=np.float32)
# 715, 180
# 570, 180
# The perspective mapping is determined. 
M = cv2.getPerspectiveTransform(pSrc,pDist)
Minv = cv2.getPerspectiveTransform(pDist,pSrc)

# Visualization of perspective mapping.
if visualizeAndSelectPerspective : 
    warped = cv2.warpPerspective(rgbCrop, M, (rgbCrop.shape[1],rgbCrop.shape[0]), flags=cv2.INTER_LINEAR)
    plt.figure()
    plt.title("cropped")
    ld.imshow_bgr(plt,rgbCrop)
    
    plt.figure()
    plt.title("cropped")
    ld.imshow_bgr(plt,rgbCrop)
    plt.plot(pSrc[:,0],pSrc[:,1])
    plt.plot(pDist[:,0],pDist[:,1])
    
    plt.figure()
    plt.title("warped")
    ld.imshow_bgr(plt,warped)
   
# Specify threholds found with the visualizeAndSelectThreshold-function.
thrSobelx = (9,255)
thrSobely = (23,255)
thrMag = (40,255)
thrDir = (145,179)
thrBwL = (132,255)
thrBwS = (132,255)
thrBwLab_B = (166,255)
thrBwHls_l = (222,255)

# Defines a mask to only include road-area in the image. 
topPositionRatio = 2.0/10
fromCenterSpreadingRatio = 2/30
dimImage = rgbCrop.shape
vertices = np.array([[(0,dimImage[0]),(dimImage[1]/2-dimImage[1]*fromCenterSpreadingRatio,dimImage[0]*topPositionRatio),(dimImage[1]/2+dimImage[1]*fromCenterSpreadingRatio,dimImage[0]*topPositionRatio),(dimImage[1],dimImage[0])]], dtype=np.int32)
mask = ld.region_of_interest(dimImage[0:2],vertices ).astype(bool)
    


if visualizeResult == 2 :
    fig1 = plt.figure()
    rect = fig1.patch
    rect.set_facecolor('white')

#dirsTestImages = [dirsTestImages[2]]
#dirsTestImages = dirsTestImages[0:3]

initWindowHeight = 0.3

if gateTestImages : 
    # Run through test images.
    for idxImage,iDirImage in enumerate(dirsTestImages): 
        print("Current image: ", idxImage+1,"/",len(dirsTestImages))
        
        # Read image
        rgb = cv2.imread(iDirImage)
        dimRgb = rgb.shape

        # Distortion correction
        camMat = calibrationData['cameraMatrix']
        distParam = calibrationData['distortionParam']
        
        # Undistortion is handled using opencv.
        rgbCor = cv2.undistort(rgb, camMat, distParam, None, camMat)    
        
        # Crop images
        imageCrop = np.round(imgCropAbs*np.array([dimRgb[0],dimRgb[0],dimRgb[1],dimRgb[1]])).astype(np.int)
        rgbCrop = rgbCor[imageCrop[0]:imageCrop[1],imageCrop[2]:imageCrop[3],:]
            
        # Visualize input image, crop and mask.
        if visualizePolyfit : 
            plt.figure()
            ld.imshow_bgr(plt,rgb)
            
            plt.figure()
            ld.imshow_bgr(plt,rgbCrop)
            
            plt.figure()
            ld.imshow_bgr(plt,cv2.bitwise_and(rgbCrop,rgbCrop,mask=mask.astype(np.uint8)))
            
        # Results of thresholding
        if useColorDetection : 
            bwCombined = ld.simpleThreshold(rgbCrop,thrBwLab_B,thrBwHls_l) 
        else : 
            bwCombined = ld.comebineGradMagDirColor(rgbCrop,thrSobelx,thrSobely,thrMag,thrDir,thrBwL,thrBwS)
        
        if visualizePolyfit and (idxImage == 6) : 
            plt.figure()
            ld.imshow_bgr(plt,rgbCrop)
            plt.savefig('./examples/rgbCropped.png', bbox_inches='tight')
            
            plt.figure()
            plt.imshow(bwCombined)
            plt.savefig('./examples/bwCombined.png', bbox_inches='tight')
        # Exclude area using a mask 
        bwCombined = bwCombined & mask
        
        # Perform perspective mapping
        bwWrapped = cv2.warpPerspective(bwCombined.astype(np.uint8), M, (bwCombined.shape[1],bwCombined.shape[0]), flags=cv2.INTER_LINEAR)    
        
        dimBw = bwWrapped.shape
        
        
        
        # Perform sliding window on.
        polyfitWindow = ld.fitLanesSlidingWindow(bwWrapped,initWindowHeight,visualizeSlidingWindow)
        
        
        polyfit = polyfitWindow.copy()
        
        # Fit polygon to lanes using an earlier polygon fit
        polyfit,curverad,vehicleShift = ld.fitLanesPoly(bwWrapped,polyfit)
        
        # Make lanes.
        nLanes = 2
        lanesX = []
        lanesY = []
        for iLane in range(nLanes) : 
            ploty = np.linspace(0, dimBw[0]-1, dimBw[0] )
            fitx = np.polyval(polyfit[iLane,:], ploty)
            lanesX.append(fitx)
            lanesY.append(ploty)
        
        # Visualize lane detection
        rgbLane = ld.visualizeLaneDetection(rgbCrop,lanesX,lanesY,Minv)
        
        # Stack zeros to rgbLane to have similar shape as the original rgb-image.
        rgbLane = np.vstack((np.zeros((rgb.shape[0]-rgbLane.shape[0],rgb.shape[1],3)),rgbLane)).astype(np.uint8)
        
        
        # Combine rgb lane and input rgb (bgr) image.
        result = cv2.addWeighted(rgb, 1, rgbLane, 0.4, 0)
        
        # Write text with curvature and vehicle shift to image. 
        result = ld.putTextCruveShift(result,curverad,vehicleShift)
    
        # Visualize polynomial fit.
        if visualizePolyfit : 
            
            plt.figure()
            plt.imshow(bwWrapped)
            plt.plot([0, dimBw[1]],[dimBw[0]*(1-initWindowHeight), dimBw[0]*(1-initWindowHeight)])
            plt.plot([dimBw[1]*0.5, dimBw[1]*0.5],[0, dimBw[0]])
            
            plt.figure()
            plt.imshow(bwWrapped)
            # fit using lanes from sliding window
            for iLane in range(nLanes) : 
                ploty = np.linspace(0, dimBw[0]-1, dimBw[0] )
                fitx = np.polyval(polyfitWindow[iLane,:], ploty)
                plt.plot(fitx,ploty,'r', linewidth=6.0)
                
            # Fit using previous polygon
            for iLane in range(nLanes) : 
                plt.plot(lanesX[iLane],lanesY[iLane],'g', linewidth=1.5)
    
        if visualizeResult == 1 : 
            plt.figure()
            ld.imshow_bgr(plt,result)
            
        if visualizeResult == 2 : 
            plt.subplot(2,4,idxImage+1)
            ld.imshow_bgr(plt,result)

        
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
cv2.namedWindow('image',cv2.WINDOW_NORMAL)

# For processing video.
if gateVideo : 
    cFrames = 0
    # Directory of input video.
    dirInputVideo = "project_video"
    #dirInputVideo = "challenge_video"
    clip1 = VideoFileClip(dirInputVideo+".mp4")
    
    print("Loading video...")
    images = [rgb for rgb  in clip1.iter_frames()]
    print("Done loading video")    
    imagesResult = []
    
    # Keeps info of previous frames.
    keepInfo = 0.7
    resetData = True
    
    polyfits = []
    #images = [images[0]]
    #images = images[500:700]
    # Step through all images in the video
    for idxImage,rgb in enumerate(images): 
        print("Current image: ", idxImage+1,"/",len(images))
        # Images are converted to BRG
        rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
        dimRgb = rgb.shape

        # Distortion correction
        camMat = calibrationData['cameraMatrix']
        distParam = calibrationData['distortionParam']
        rgbCor = cv2.undistort(rgb, camMat, distParam, None, camMat)    
        
        # Crop image
        imageCrop = np.round(imgCropAbs*np.array([dimRgb[0],dimRgb[0],dimRgb[1],dimRgb[1]])).astype(np.int)
        rgbCrop = rgbCor[imageCrop[0]:imageCrop[1],imageCrop[2]:imageCrop[3],:]
            
    
        # Threshold using sobelx,sobely,gradient magnitude, gradient directions, l (hls) and s(hls)
        #bwCombined = ld.comebineGradMagDirColor(rgbCrop,thrSobelx,thrSobely,thrMag,thrDir,thrBwL,thrBwS)
        # Results of thresholding
        if useColorDetection : 
            bwCombined = ld.simpleThreshold(rgbCrop,thrBwLab_B,thrBwHls_l) 
        else : 
            bwCombined = ld.comebineGradMagDirColor(rgbCrop,thrSobelx,thrSobely,thrMag,thrDir,thrBwL,thrBwS)
            
        # Include only road area using a mask. 
        bwCombined = (bwCombined & mask).astype(np.float16)
        
        # Lane data is reset.
        if resetData:
            rememberMask = np.zeros_like(bwCombined)
            rememberMask = bwCombined*255
        else:
            # A mask for rememberings pervious lane detections
            rememberMask = np.minimum(rememberMask*(1-keepInfo)+bwCombined*255,255.0)
        
        # Perspective mapping using opencv function.
        bwWrapped = cv2.warpPerspective(rememberMask.astype(np.uint8), M, (bwCombined.shape[1],bwCombined.shape[0]), flags=cv2.INTER_LINEAR)    
        
        dimBw = bwWrapped.shape
        #rgbWrapped = np.dstack((bwWrapped,bwWrapped,bwWrapped))*255
        
        
        if resetData : 
            print("Initially lanes are founding using sliding window")
            polyfitWindow = ld.fitLanesSlidingWindow(bwWrapped,initWindowHeight,visualizeSlidingWindow)
            polyfit = polyfitWindow.copy()
            resetData = False
        
        # Fit polygon to lanes using an earlier polygon fit
        polyfit,curverad,vehicleShift = ld.fitLanesPoly(bwWrapped,polyfit)
        
        
        polyfitDiff = np.squeeze(np.diff(polyfit,axis=0))
        polyfits.append(polyfitDiff)
        
        # Lane data is reset if bend - the first polynomial coefficient - is to different between lanes. 
        if np.abs(polyfitDiff[0])>0.001 : 
            print("Lane data is reset.")
            resetData = True
            
        # Polynomial lines. 
        nLanes = 2
        lanesX = []
        lanesY = []
        for iLane in range(nLanes) : 
            ploty = np.linspace(0, dimBw[0]-1, dimBw[0] )
            fitx = np.polyval(polyfit[iLane,:], ploty)
            lanesX.append(fitx)
            lanesY.append(ploty)
        
        # Visualize lane detection
        rgbLane = ld.visualizeLaneDetection(rgbCrop,lanesX,lanesY,Minv)
        
        # Stack zeros to rgbLane to have similar shape as the original rgb-image.
        rgbLane = np.vstack((np.zeros((rgb.shape[0]-rgbLane.shape[0],rgb.shape[1],3)),rgbLane)).astype(np.uint8)
        
        
        # Combine rgb lane and input rgb (bgr) image.
        result = cv2.addWeighted(rgb, 1, rgbLane, 0.4, 0)
        
        # Write text with curvature and vehicle shift to image. 
        result = ld.putTextCruveShift(result,curverad,vehicleShift)
        
        # Append image to list in rgb format, This list is converted to a video
        imagesResult.append(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
        
        if resetData or (cFrames>0):
            result = ld.putTextReset(result)
            cFrames = cFrames+1
            if cFrames == 6 : 
                cFrames = 0
        cv2.imshow('image',result)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    
    # Create video from images. 
    new_clip = ImageSequenceClip(imagesResult, fps=clip1.fps)
    new_clip.write_videofile(dirInputVideo + "_Processed.mp4") 
    
    polyfits_np = np.squeeze(np.array(polyfits))
    plt.figure()
    plt.plot(polyfits_np[:,0])