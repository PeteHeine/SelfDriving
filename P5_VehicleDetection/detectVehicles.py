#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 08:39:08 2017

@author: repete
"""

import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    heatmap = np.zeros((draw_img.shape[0],draw_img.shape[1]))
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
#b = a,0)
            #print("hog_features.shape",hog_features.shape,"spatial_features.shape",spatial_features.shape,"hist_features.shape",hist_features.shape)
            featuresStacked = np.hstack((hog_features, hist_features,spatial_features))
            
            #print("featuresStacked.shape",featuresStacked.shape)
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(featuresStacked.reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox = (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)
                #print(heatmap.shape)
                heatmap[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]] += 1
                cv2.rectangle(draw_img,bbox[0],bbox[1],(0,0,255),6) 
            
                
    return draw_img,heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
    
#def bin_spatial(img, size=(32, 32)):
#    color1 = cv2.resize(img[:,:,0], size).ravel()
#    color2 = cv2.resize(img[:,:,1], size).ravel()
#    color3 = cv2.resize(img[:,:,2], size).ravel()
#    return np.hstack((color1, color2, color3))
#    
## Define a function to return HOG features and visualization
#def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
#    # Call with two outputs if vis==True
#    if vis == True:
#        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
#                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
#                                  visualise=vis, feature_vector=feature_vec)
#        return features, hog_image
#    # Otherwise call with one output
#    else:      
#        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
#                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
#                       visualise=vis, feature_vector=feature_vec)
#        return features
#
#def color_hist(img, nbins=32):    #bins_range=(0, 256)
#    # Compute the histogram of the color channels separately
#    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
#    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
#    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
#    # Concatenate the histograms into a single feature vector
#    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
#    # Return the individual histograms, bin_centers and feature vector
#    return np.ravel(hist_features)
    
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, nbins=32,spa_features=32,include_features='111'):
    # Create a list to append feature vectors to
    features = []
    print("include_features",include_features)
    featureShapes = []
    # Iterate through the list of images
    for idxfile,file in enumerate(imgs):
        # Read in each one by one
        image = mpimg.imread(file)
        allFeatures = np.array([])
        if bool(int(include_features[0])) : 
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif cspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
                elif cspace == 'LAB':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            else: feature_image = np.copy(image)      
    
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            allFeatures = np.hstack((allFeatures,hog_features))
            if idxfile == 0 : 
                print("hog_features.shape",hog_features.shape)
                featureShapes.append(hog_features.shape)
                
        if bool(int(include_features[1])): 
            color_features = color_hist(image,nbins = nbins)
            allFeatures = np.hstack((allFeatures,color_features))
            if idxfile == 0 : 
                print("color_features.shape",color_features.shape)
                featureShapes.append(color_features.shape)
        
        if bool(int(include_features[2])) : 
            spatial_features = bin_spatial(image,(spa_features,spa_features))
            allFeatures = np.hstack((allFeatures,spatial_features))
            if idxfile == 0 : 
                print("spatial_features.shape",spatial_features.shape)
                featureShapes.append(spatial_features.shape)
        if idxfile == 0 : 
            print("allFeatures.shape",allFeatures.shape)
            featureShapes.append(allFeatures.shape)
        #allFeatures = np.hstack((spatial_features, color_features, hog_features))
        # Append the new feature vector to the features list
        features.append(allFeatures)
    # Return list of feature vectors
    return features,featureShapes