#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:42:43 2017

@author: repete
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function for taking a list of checkerboard images directories for calibrating camera.
def myCameraCalibration(dirs_calibration_images,checkerboard_grid) : 
    # Number of calibration images.
    nImages = len(dirs_calibration_images)
    
    # Create object points (real work coordinates)
    objPoints = np.zeros((checkerboard_grid[0]*checkerboard_grid[1],3),np.float32)
    objPoints[:,:2] = np.mgrid[0:checkerboard_grid[0],0:checkerboard_grid[1]].T.reshape(-1,2)
    
    checkboardCorners = []
    checkboardObjCorners = []
    checkboardDetected = []
    # Iterate through all calibration images.
    for idxImage, iDirImage in enumerate(dirs_calibration_images) : 
        print("Detect checkerboards: ", idxImage+1,"/",nImages)
        # Read images as rgb
        rgb = cv2.imread(iDirImage)
        
        # Convert image to gray scale (monochromatic) 
        img = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        
        # Detect checkerboard
        ret, tmpCorners = cv2.findChessboardCorners(img, checkerboard_grid, None)

        # Append valid checker boards.
        checkboardDetected.append(ret)
        if ret : 
            # Append detect checkerboard corners
            checkboardCorners.append(tmpCorners)
            # Append object points. These are identical for each image.  
            checkboardObjCorners.append(objPoints)
    
    # Camera is calibrated
    ret, cameraMatrix, distortionParam, cbRotation, cbTranslation= cv2.calibrateCamera(checkboardObjCorners, checkboardCorners, img.shape[::-1], None, None)

    calibrationData = {}
    calibrationData['distortionParam'] = distortionParam
    calibrationData['cameraMatrix'] = cameraMatrix    
    return calibrationData

# simple helper function to show bgr image using matlibplot.pyplot
# Typical usage: 
    # import matplotlib.pyplot as plt
    # plt.figure()
    # imshow_bgr(plt,bgrImage)
def imshow_bgr(obj,brg) : 
    obj.imshow(cv2.cvtColor(brg,cv2.COLOR_BGR2RGB))

# Sobel in x and y direction.
def performSobel(rgb,sobel_kernel=3) : 
    # Apply the following steps to img
    # 1) Convert to grayscale
    img = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    
    # 2) Determine the gradient in x and y separately
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobel_kernel)
    return sobelx,sobely
        

# Absolute sobel. 
def abs_sobel(sobelfilt) :

    # 1) Take the absolute value of the derivative or gradient
    abs_sobelfilt = np.absolute(sobelfilt)
    
    # 2) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelfilt/np.max(abs_sobelfilt))
    
    return scaled_sobel

# Magnitude of gradients. 
def magnitude(sobelx,sobely) :
    
    # 1) Calculate the magnitude 
    abs_sobelxy = np.sqrt(sobelx**2+sobely**2)
    
    # 2) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    gradmag = (255.0*abs_sobelxy/np.max(abs_sobelxy)).astype(np.uint8)
    
    return gradmag
# Direction of gradient    
def sobelDirections(sobelx,sobely):
    
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    orientation = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
    orientation = 127.5*(orientation+np.pi)/np.pi # between 0 and 255
    
    return orientation

# Calculate HLS.
def hls(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2HLS)

def lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
# Dummie function.
def nothing(x):
        pass

# Function to visualize the selected thresholds. 
def visualizeAndSelectThreshold(rgbStacked,selectType):
    # Sobelx, sobely, magnitude, gradient directions and the color transform is 
    # only calculated once. 
    sobelx,sobely = performSobel(rgbStacked)
    sobelx_abs = abs_sobel(sobelx)
    sobely_abs = abs_sobel(sobely)
    mag = magnitude(sobelx,sobely)
    direction = sobelDirections(sobelx,sobely)
    hlsColors = hls(rgbStacked)
    labColors = lab(rgbStacked)
        
    # selectType = 0: To visualize threshold of sobelx and sobely
    if selectType == 0: 
        trackerNames = ['sobelx','sobely']
        valImages = [sobelx_abs,sobely_abs]
    
    # selectType = 1: To visualize threshold of Magnitude and direction of gradients. 
    if selectType == 1: 
        trackerNames = ['Magnitude','Direction']
        valImages = [mag,direction]
    
    # selectType = 2: To visualize threshold l and s in the hls color space. 
    if selectType == 2: 
        trackerNames = ['l','s']
        valImages = [hlsColors[:,:,1],hlsColors[:,:,2]]

    if selectType == 3: 
        trackerNames = ['lab_b','hls_l']
        valImages = [labColors[:,:,2,],hlsColors[:,:,1]]
    # Make window.
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    
    # Convert to gray.
    imgStacked = cv2.cvtColor(rgbStacked,cv2.COLOR_BGR2GRAY)
    outputStacked = np.copy(imgStacked)
    # Create a min and max trackbar for each trackerName (e.g. magnitude)
    for iTrackerName in trackerNames : 
        cv2.createTrackbar(iTrackerName+'Min','image',0,255,nothing)
        cv2.createTrackbar(iTrackerName+'Max','image',0,255,nothing)
        # Stack empty image. These will later show the thresholded values.
        outputStacked = np.vstack((outputStacked,np.zeros_like(imgStacked,dtype=np.uint8)))
    
    
    currentTrackerPositionNew = np.zeros((len(trackerNames),2))
    currentTrackerPositionOld = np.ones((len(trackerNames),2))
    # Continues until the "Esc" character is being pressed
    while(1):
        # Show stacked image 
        cv2.imshow('image',outputStacked)
        
        # Wait for key
        k = cv2.waitKey(1) & 0xFF
        # Break-loop on esc.
        if k == 27:
            break
    
        # Get current position of min and max tracker for a given tracker name.
        for idxTrackerName, iTrackerName in enumerate(trackerNames): 
            currentTrackerPositionNew[idxTrackerName,0] = cv2.getTrackbarPos(iTrackerName+'Min','image')
            currentTrackerPositionNew[idxTrackerName,1] = cv2.getTrackbarPos(iTrackerName+'Max','image')
            
        # Do nothing if nothing has changed. 
        if np.array_equal(currentTrackerPositionNew,currentTrackerPositionOld): 
            pass # Do nothing
        else: 
            print("Registered changes")
            currentTrackerPositionOld = np.copy(currentTrackerPositionNew)
            outputStacked = np.copy(imgStacked)
            bwAnd = np.zeros_like(imgStacked)
            for idxTracker, iTrackerName in enumerate(trackerNames): 
                # Use the trackerbar specied tresholds.
                bw = np.zeros_like(valImages[idxTracker],dtype=np.uint8)
                bw[(valImages[idxTracker]>currentTrackerPositionNew[idxTracker,0]) & (valImages[idxTracker]<currentTrackerPositionNew[idxTracker,1])] = 255
                
                if idxTracker == 0:
                    bwAnd = bw
                else : 
                    bwAnd = (bwAnd & bw)
                # Stack the thresholds 
                outputStacked = np.hstack((outputStacked,bw))
            outputStacked = np.hstack((outputStacked,bwAnd))
    
    cv2.destroyAllWindows()
    
# Apply upper max and a min treshold.    
def applyThreshold(img,thresh) : 
    bw = np.zeros_like(img,dtype=bool)
    bw[(img>thresh[0])&(img<thresh[1])] = True
    return bw

# Gradient, magnitude, direction and colors     
def comebineGradMagDirColor(rgbCrop,thr_sobelx,thr_sobely,thr_mag,thr_dir,thr_bwL,thr_bwS) : 
    sobelx,sobely = performSobel(rgbCrop)
    bwSobelx= applyThreshold(abs_sobel(sobelx),thr_sobelx)
    bwSobely= applyThreshold(abs_sobel(sobely),thr_sobely)
    bwMag= applyThreshold(magnitude(sobelx,sobely),thr_mag)    
    bwDirection= applyThreshold(sobelDirections(sobelx,sobely),thr_dir)
    
    hlsColors = hls(rgbCrop)
    bwL= applyThreshold(hlsColors[:,:,1],thr_bwL)    
    bwS= applyThreshold(hlsColors[:,:,2],thr_bwS)
    
    return (bwSobelx & bwSobely) | (bwMag & bwDirection) | (bwL & bwS)

def simpleThreshold(rgbCrop,thrBwLab_B,thrBwHls_l) : 
    hlsColors = hls(rgbCrop)
    labColors = lab(rgbCrop)
    return applyThreshold(hlsColors[:,:,1],thrBwHls_l) | applyThreshold(labColors[:,:,2],thrBwLab_B)
# Function for finding lanes using the sliding window function.
def fitLanesSlidingWindow(bwWrapped,initWindowHeight,doPlotting) :
    
    ## Ignore border areas for detecting the first position for left and right lane.
    removeBorderArea = 300
    
    ## Sliding window
    # Number of windows
    nWindows = 8
    # Height of window
    windowHeight = int(bwWrapped.shape[0]/nWindows)
    # window margin
    windowMargin = 100
    
    minpix = 100
    # Reposition window for every 
    reposition = True
    
    nLanes = 2 
    
    
    ## Poly fit
    nOrders = 2
    useWeightning = True
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bwWrapped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Stack mask to r,g and b to be able to draw with color (later). 
    rgbWrapped = np.dstack((bwWrapped,bwWrapped,bwWrapped))*255
    dimBw = bwWrapped.shape
    
    # Center position of image.
    centerCar = np.round(bwWrapped.shape[1]*0.5).astype(np.int)
    
    # Top position of window
    initWindow = np.round(bwWrapped.shape[0]*(1-initWindowHeight)).astype(np.int)
        
    # Make histogram to detect lanes.
    colHist = np.sum(bwWrapped[initWindow:,:],axis=0)
    
    # Set border areas to zero using removeBorderArea-variable.
    colHist[0:removeBorderArea] = 0
    colHist[-removeBorderArea:] = 0

    # Find peak position of histogram to detect left and right lane.
    lanePeaks = [np.argmax(colHist[:centerCar]),centerCar+np.argmax(colHist[centerCar:])]

    # Do plotting.
    if doPlotting : 
        plt.figure()
        plt.plot(colHist)
        plt.plot([centerCar,centerCar],[0,np.max(colHist)])
        plt.plot(lanePeaks,colHist[lanePeaks],'.')
    
    # Define lane position.
    pBase = np.array(lanePeaks)
    
    # Append these indices to the lists
    lane_idx = [[] for _ in range(2)]
                
    
    pixelsInWindows = np.zeros((nWindows,nLanes))
    pixelsInWindow = np.zeros((nLanes))
    pBaseAll = np.zeros((nWindows,nLanes))
    
    # Step through each window
    for iWindow in range(nWindows) :                       
        
        ## Specify windowCoordinats 
        # Y-Coordinates
        pTop = np.ones((2,1))*(dimBw[0]-(iWindow)*windowHeight)
        pBottom = pTop-windowHeight
        
        # X-Coordinates
        pLeft = pBase-windowMargin
        pRight = pBase+windowMargin
        
        # Step through the two lanes.
        for iLane in range(nLanes) : 

            # Get idx of in window coordinates.
            inWindowIdx = ((nonzerox >= pLeft[iLane] ) & (nonzerox < pRight[iLane]) & (nonzeroy < pTop[iLane] ) & (nonzeroy >= pBottom[iLane])).nonzero()[0]
            
            # Number of points in a window.
            pixelsInWindow[iLane] = len(inWindowIdx)
            
            # Use pervious window position, when no pixels are in the window
            if len(inWindowIdx) < 1 :
                pBase[iLane] = pBase[iLane]
            # Use the mean x-position of in-window pixels.
            else : 
                pBase[iLane] = np.mean(nonzerox[inWindowIdx])
            
            pBaseAll[iWindow,iLane] = pBase[iLane]

            # Reposition sliding window.             
            if reposition :
                pLeft[iLane] = pBase[iLane]-windowMargin
                pRight[iLane] = pBase[iLane]+windowMargin
                inWindowIdx = ((nonzerox >= pLeft[iLane] ) & (nonzerox < pRight[iLane]) & (nonzeroy < pTop[iLane] ) & (nonzeroy >= pBottom[iLane])).nonzero()[0]
                if len(inWindowIdx) < 1 :
                    pBase[iLane] = pBase[iLane]
                else : 
                    pBase[iLane] = np.mean(nonzerox[inWindowIdx])

            # Append window indices.
            lane_idx[iLane].append(inWindowIdx)
        
        pixelsInWindows[iWindow,:] = pixelsInWindow
        
        validLane = pixelsInWindow>minpix
        
        # A lane detection is ignored for too few pixels.
        if (iWindow>0) and (validLane[0] == True) and (validLane[1] == False):
            pBase[1] = pBaseAll[iWindow-1,1]+np.diff(pBaseAll[iWindow-1:iWindow+1,0])
        if (iWindow>0) and (validLane[0] == False) and (validLane[1] == True):
            pBase[0] = pBaseAll[iWindow-1,0]+np.diff(pBaseAll[iWindow-1:iWindow+1,1])
        
        # Visualize lane and the sliding window approach.
        for iLane in range(nLanes) : 
            # Visualize rectangles 
            cv2.rectangle(rgbWrapped,(pLeft[iLane],pTop[iLane]),(pRight[iLane],pBottom[iLane]),(0,255,0),4)
            cv2.circle(rgbWrapped, (pBase[iLane],pTop[iLane]-windowHeight/2), 8, (255,0,0),thickness=5)

    # Perform sanity check
    np.diff(pBaseAll,axis=0)
    
        
    if(doPlotting) :
        plt.figure()
        imshow_bgr(plt,rgbWrapped)
    
    # Fit a polynomial curve to tracks.
    polyfit = np.zeros((nLanes,nOrders+1))
    for iLane in range(nLanes) : 
        lane_idx_all = np.concatenate(lane_idx[iLane])
        pX = nonzerox[lane_idx_all]
        pY = nonzeroy[lane_idx_all]
        
        if useWeightning : 
            # All points are weighted relative to the row-position...
            # Meaning that lane points close to the car are weighted higher (high row position) and
            # and positions far away are weighted less... No point is weighted less  than 30.
            pWeight = np.maximum(pY,30) 
        else :
            pWeight = np.ones_like(pY)
        # Fit lane to polynomial.
        polyfit[iLane,:] = np.polyfit(pY,pX, nOrders,w=pWeight)
        
        ploty = np.linspace(0, dimBw[0]-1, dimBw[0] )
        fitx = np.polyval(polyfit[iLane,:], ploty)
        if(doPlotting) :
            plt.plot(fitx,ploty)
            
    #return polyfit
    return polyfit

# Visualize: Lane detection.    
def visualizeLaneDetection(rgbCrop,lanesX,lanesY,Minv):
    
    # Create an image to draw the lines on
    #warp_zero = np.zeros_like(bwWrapped).astype(np.uint8)
    dimBw = rgbCrop.shape[0:2]
    warp_zero = np.zeros(dimBw).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lanesX[0],lanesY[0]]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([lanesX[1],lanesY[1]])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image (bgr)
    cv2.fillPoly(color_warp, np.int_([np.squeeze(pts)]), (0,255, 0))
    
#    leftLine = np.transpose(np.array([lanesX[0],lanesY[0]]))
#    rightLine = np.transpose(np.array([lanesX[1],lanesY[1]]))
#    cv2.polylines(color_warp, np.int_([leftLine]) , 0,(255,0,0),thickness=10)
#    cv2.polylines(color_warp, np.int_([rightLine]), 0,(255,0,0),thickness=10)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    return cv2.warpPerspective(color_warp, Minv, (dimBw[1], dimBw[0])) 

# Make mask from vertices.
def region_of_interest(img_shape, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros(img_shape)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img_shape) > 2:
        channel_count = img_shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    return cv2.fillPoly(mask, vertices, ignore_mask_color)
    
# Insert text of vehicle shift and lane curvature in image.
def putTextCruveShift(rgb,curverad,vehicleShift) : 
    lineThickness = 4
    fontSize = 2
    print("curverad",curverad,"mean",np.mean(curverad))
    textColor = (255,255,255)
    text1 =  "Curvature: " + str(np.round(np.mean(curverad))) + "m"
    text2 =  "Vehicle shift: " + str(np.round(vehicleShift,decimals=2)) + "(m)"
    rgb = cv2.putText(rgb,text1, (100,100), cv2.FONT_HERSHEY_SIMPLEX, fontSize, textColor, lineThickness)
    rgb = cv2.putText(rgb,text2, (100,170), cv2.FONT_HERSHEY_SIMPLEX, fontSize, textColor, lineThickness)
    return rgb

# Insert text. 
def putTextReset(rgb) : 
    lineThickness = 6
    fontSize = 2
    textColor = (255,0,0)
    text1 =  "Adjust lanes"    
    rgb = cv2.putText(rgb,text1, (10,int(rgb.shape[0]-10)), cv2.FONT_HERSHEY_SIMPLEX, fontSize, textColor, lineThickness)
    return rgb

# Fit lanes using the already defined polynomial lines. 
def fitLanesPoly(bwWrapped,polyfit) : 
    nonzero = bwWrapped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Margin to be included.
    margin = 100
    nOrders = 2
    nLanes = 2
    
    # Use weights for fitting.polynomial. 
    useWeightning = True

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 6/30 # meters per pixel in y dimension
    xm_per_pix = 3.7/450 # meters per pixel in x dimension
    
#    ym_per_pix = 30/720 # meters per pixel in y dimension
#    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    curverad = np.zeros(2)
    for iLane in range(nLanes) :
        # fit polynomial function to lane
        fitx = np.polyval(polyfit[iLane,:], nonzeroy)
        
        # Find point within the polynomial fit with a margin.
        inMarginIdx = (nonzerox>(fitx-margin)) & (nonzerox<(fitx+margin))
        
        # Get x and y coordinates.
        pX = nonzerox[inMarginIdx]
        pY = nonzeroy[inMarginIdx]
        
        y_eval = np.max(pY)
        
        if useWeightning : 
            # All points are weighted relative to the row-position...
            # Meaning that lane points close to the car are weighted higher (high row position) and
            # and positions far away are weighted less... No point is weighted less  than 30.
            pWeight = np.maximum(pY,30) 
        else :
            pWeight = np.ones_like(pY)
            
        polyfit[iLane,:] = np.polyfit(pY,pX, nOrders,w=pWeight)
        
    
        # Determine curvature    
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(pY*ym_per_pix, pX*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        curverad[iLane] = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    
    pLane = np.zeros((2))
    for iLane in range(nLanes) :
        pLane[iLane] = np.polyval(polyfit[iLane,:], bwWrapped.shape[0])
    
    # Get vehicle shift in meters. 
    vehicleShift = np.mean(pLane)-bwWrapped.shape[1]/2
    vehicleShiftInM = vehicleShift*xm_per_pix
    #print(curverad[0], 'm', curverad[1], 'm')
    return polyfit,curverad,vehicleShiftInM