from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense, Conv2D
import numpy as np
import cv2
import sys
import gradio as gr
import skimage.segmentation
#import everything why not

#function to take in a greyscale 3x3 subimage and return the correct LBP label for that pixel
def getOneLBPLabel(subimage, label_type):
    label = 0
    #neighborhood storage
    neighborhood = []
    #binary number list
    binaryNumber = []
    
    #i love hardcoding
    neighborhood.append([subimage[1][0]])
    neighborhood.append([subimage[2][0]])
    neighborhood.append([subimage[2][1]])
    neighborhood.append([subimage[2][2]])
    neighborhood.append([subimage[1][2]])
    neighborhood.append([subimage[0][2]])
    neighborhood.append([subimage[0][1]])
    neighborhood.append([subimage[0][0]])

    #thresholding
    for i in neighborhood:
        if i <= subimage[1][1]:
            binaryNumber.append(0)
        else:
            binaryNumber.append(1)
    #changes
    M = 0
    bitsWithOne = 0
    #iterate through binary and check for changes without checking the out of bounds
    for i in range(len(binaryNumber) - 1):
        current = binaryNumber[i]
        next = binaryNumber[i + 1]
        if current != next:
            M = M + 1
    #if M is less than 2 label is number of 1s in binary
    if M <= 2:
        for i in binaryNumber:
            if i == 1:
                bitsWithOne = bitsWithOne + 1
        label = bitsWithOne
    #if M is greater than 2 M is length of binary + 1
    elif M > 2:
        label = len(binaryNumber) + 1
    #returns the label
    return label 
       
def getLBPImage(image, label_type):
    #set the radius, ig i could've just used 1
    radius = 1
    #copying and padding the image
    labelImage = np.copy(image)
    #used border constant and value = 0 since they kinda seemed like defaults and it yelled at me for
    #having no border, pads the image
    paddedImage = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    #grab height and width from the image, height is 0 width is 1 learned it the hard way
    height = image.shape[0]
    width = image.shape[1]

    #loop through each pixel, adding the radius to account for padding
    for i in range(radius, height + radius):
        for j in range(radius, width + radius):
            #grabs from i - radius to i + radius because python excludes the right side call of i+radius+1
            #for some reason, took me half an hour to figure this out.
            #i liked the code we used in A02 that did this so i figured it out here
            subimage = paddedImage[i - radius:i + radius + 1, j - radius:j + radius + 1]

            #gets the label and sets the label in the new image
            LBPLabel = getOneLBPLabel(subimage, label_type)
            #you have to subtract radius here because of the padding to get position in original image
            labelImage[i - radius][j - radius] = LBPLabel
    #returns the label image    
    return labelImage

def getOneRegionLBPFeatures(subImage, label_type):
        #calculates histogram using the appropriate size for uniform
        hist = cv2.calcHist([subImage], [0], None, [10], [0, 10])
        #normalizes the histogram
        hist =  hist / np.sum(hist)
        #returns the histogram
        #it was returned flattened in A01 so why not here, also it didnt work unflattened
        return hist.flatten()

def getLBPFeatures(featureImage, regionSideCnt, label_type):
    #list to store hists
    hists = []
    
    #height and width, thank you A02, height is 0 width is 1 learned that the hard way
    subHeight = featureImage.shape[0] // regionSideCnt
    subWidth = featureImage.shape[1] // regionSideCnt
    #looping through each subregion
    #explained in class 11/20/23, starting point is i/j * subheight/subwidth
    for i in range(regionSideCnt):
        for j in range(regionSideCnt):      
            subImage = featureImage[i * subHeight: i * subHeight + subHeight, j * subWidth: j * subWidth + subWidth]
            #get hist
            hist = getOneRegionLBPFeatures(subImage, label_type)
            #append hist
            hists.append(hist)
    #change hists to np array
    hists = np.array(hists)
    #reshape hists array
    hists = np.reshape(hists, (hists.shape[0] * hists.shape[1]))
    #return hists
    return hists
            
            
    
        

    
        