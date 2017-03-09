import pandas as pd 
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import os
import matplotlib.pyplot as plt
from scipy import misc

### Experiments 
# 1) Using only center images work better 
# 2) Udacity data doesn't improve ?
# 3) A wider network doesn't seem to work. (24 in deeper layers)
# 4) Using a bigger batch size (around 100 and too big). Might have improved accuracy.
# 5) Dropout? Not really working.
# 6) Adding another fully connected layer (a total of three fc)? No
# 7) Using longer training (10 epochs)? No. Use five????
# 8) USE VALID PADDING !!!!!!
# 9) Increasing steering correction from 0.10 to 0.27... YEAHHH it worked
# 10) Crop top and bottom of image? Overshoots with steering correcion 0.27. NOT WORKING
# 11) Dropout? Not working

### Load data ##############################################################
# Bool to select remote or host computing.
useRemote = False
if useRemote == True:
    dirDataOut = "/media/slow-storage/DataFolder/Peter/SelfDriving/CollectedSets"
else:
    dirDataOut = "/home/repete/DataFolder/SelfDriving/CollectedSets"

# The number of epochs used in training.
nEpochs = 8

# Parameters for selecting subsets of the data.
# The image and csv-files have been pre-generated have been generated in a another script. 
# The scripts resamples images by 2 and I have also smoothened steering angles (I did not used a joystick) 
selectedNames = sorted(['peter','mikkel','udacity']) # Data to be included.
smoothSteering = sorted(['peter','mikkel','udacity'])
# An image resizing of 2 have been used.
imageResizing = 2
# Steerings angles have been smoothened with a average kernel of nSize.
nSize = 5

# Loads a specific data-folder based on above specification. 
inputName = "Resizing" + str(imageResizing) + "_AddedData" + ''.join(selectedNames) + '_SmoothSteeringAmount' + str(nSize) + "To" +''.join(smoothSteering)
# Dir of images.
dirInputImages = os.path.join(dirDataOut,inputName,"IMG")
# Dir of cvs-files with data.
data = pd.read_csv(os.path.join(dirDataOut,inputName,'driving_log.csv'))
# Select batch size.
batchSize = 100

### Split data into train, validation and test set.
# Random indices are generated to select random data for train, validation and test set.
randIndex = np.random.permutation(data.shape[0])
# Select the amount of data used for validation and test set. 
pVal = 0.1
pTest = 0.1
dataTrain = data.reindex([randIndex[0                           :data.shape[0]*(1-pVal-pTest)]])
dataVal = data.reindex(  [randIndex[data.shape[0]*(1-pVal-pTest):data.shape[0]*(1-pTest)]])
dataTest = data.reindex( [randIndex[data.shape[0]*(1-pTest)     :]])

print("Total number of samples: ",data.shape[0]) # Number of training samples:  12562            
print("Number of training samples: ",dataTrain.shape[0]) # Number of training samples:  12562
print("Number of validation samples: ",dataVal.shape[0]) # Number of validation samples:  1570
print("Number of test samples: ",dataTest.shape[0])      # Number of test samples:  1571

# A python generator is used for selecting batches used in training, validation and test.
# Bools specify wether to use normalization, mirroring and left/right cameras.
def generatorGetBatch(data,dirInputImages,batchSize,normalizePerImage=True, plotting=False, doMirror = True, useLeftRight = True):
    # SteeringCorrection. 
    # a) Added to the steeringAngle, when the left camera is used. 
    # b) Subtracted from the steeringAngle, when the left camera is used. 
    steeringCorrection = 0.27
    
    # Bool to specify if crop top and bottom of input images.
    # This haven't improve performanced and therefore not included. 
    doCrop = False 
    
    imgNames = ['center','left','right']
    # The while-True loop ensured that the generator will never run out of samples. 
    while True:
        # Data is shuffled for every epoch.
        dataShuffled = data.reindex(np.random.permutation(data.index))
        
        # For every next()-call a batch is collected and returned using the yield-cmd.
        for iBatch in range(0,data.shape[0],batchSize):
            # Lists for appending images (X) and steeringsangles (y) to a batch. 
            X = []
            y = []
            augType = []

            # Create a batch
            dfBatch = dataShuffled.iloc[iBatch:iBatch+batchSize]            
            
            # Run through all samples in the batch. 
            for _, out in dfBatch.iterrows():                
                # Steering angle
                steering = out['steering']
                
                # Use left and right camera.
                if useLeftRight:
                    # center, left or right is selected at random.
                    augmentationType = imgNames[np.random.randint(3)]
                    # Steering correction is perform for only left and right camera.
                    if(augmentationType=="left"):
                        steering = steering+steeringCorrection
                    if(augmentationType=="right"):
                        steering = steering-steeringCorrection
                else:
                    # Only center camera is used. imgName[0] = 'center'
                    augmentationType = imgNames[0] # Use always center.

                # The image directory                
                dirImg = out[augmentationType]
                dirImg = os.path.join(dirInputImages,dirImg.split("/")[-1])

                # Reads img.
                img = misc.imread(dirImg)
                
                # If enabled: Visualize image. 
                if(plotting):
                    plt.figure()
                    plt.imshow(img)
                
                # If enabled: Performs normalization per-image
                if(normalizePerImage):
                    mu = np.mean(np.reshape(img,(-1,3)),axis=0)
                    sig = np.sqrt(np.var(np.reshape(img,(-1,3)),axis=0))
                    img = (img-mu)/sig
                
                # If enabled: Img and steering angle is mirrored half the time.
                if doMirror and np.random.randint(2)==0:
                    #mCount = mCount+1
                    #print(mCount)
                    steering = -steering
                    img = np.fliplr(img)
                
                # If enabled: Crops top (sky) and bottom (car) of the image.
                if doCrop :
                    pTopCrop  = 0.2
                    topCrop = img.shape[0]*pTopCrop
                    pBottomCrop  = 0.1
                    bottomCrop = img.shape[0]-img.shape[0]*pBottomCrop
                    print("Crop")
                    img = img[topCrop:bottomCrop,:,:]
                # Image and steering is appended to lists
                X.append(img)
                y.append(steering)
                augType.append(augmentationType)
            # Images and steering is converted into a numpy-array and returned.
            yield np.array(X), np.array(y)

            
# Test and visualize for a single batch 
test = generatorGetBatch(data,dirInputImages,10,True,False,True,True) 
#test = overFitBatch(data,10,True,True )
for iBatch in range(0,1):
    Xtest,ytest = next(test)

dimX = Xtest.shape[1:4]
img = Xtest[0,:,:,:]


### Create network #########################################################
if(False):
    
    
    
    model = Sequential()
    # Valid padding have been used for all convolutions. 
    padType = 'valid'  
    
    nKernels = 12
    kernelSize = 5
    # Apply a 5x5 convolution with 12 output filters on input image
    model.add(Convolution2D(nKernels, kernelSize, kernelSize, border_mode=padType, input_shape=(dimX)))
    model.add(Activation('relu'))
    
    # Apply max-pooling for subsamling the image.
    model.add(MaxPooling2D((2, 2)))
    
    nKernels = 24
    kernelSize = 5
    # Apply a 5x5 convolution with 24 output filters on input image
    model.add(Convolution2D(nKernels, kernelSize, kernelSize, border_mode=padType))
    model.add(Activation('relu'))
    
    # Apply max-pooling for subsamling the image.
    model.add(MaxPooling2D((2, 2)))
    
    nKernels = 24
    kernelSize = 3
    # Apply a 3x3 convolution with 24 output filters on input image
    model.add(Convolution2D(nKernels, kernelSize, kernelSize, border_mode=padType))
    model.add(Activation('relu'))
    
    # Apply max-pooling for subsamling the image.
    model.add(MaxPooling2D((2, 2)))
    
    nKernels = 24
    kernelSize = 3
    # Apply a 3x3 convolution with 24 output filters on input image
    model.add(Convolution2D(nKernels, kernelSize, kernelSize, border_mode=padType))
    model.add(Activation('relu'))
    
    # Apply max-pooling for subsamling the image.
    model.add(MaxPooling2D((2, 2)))
    
    # Feature map returned for the fully convolutional part of the network is flattened into a 1D-vector.
    model.add(Flatten())
    
    # A fully connected layer with 200 neurons/kernels with dropout.
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    # A fully connected layer with 200 neurons/kernels with dropout.
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    # The output must be a single value (a single neuron) to control the vehicle. 
    model.add(Dense(1))
    
    # The mean-squared-error is used as a loss-function with an adam-optimizer.
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    
    # fit_generator is used for training the model.
    model.fit_generator(generatorGetBatch(dataTrain,dirInputImages,batchSize,True,False,True,True),samples_per_epoch=dataTrain.shape[0], nb_epoch=nEpochs,validation_data=generatorGetBatch(dataTest,dirInputImages,batchSize,True,False,True,True),nb_val_samples=dataTest.shape[0])
    
    ### Model is stored-
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
