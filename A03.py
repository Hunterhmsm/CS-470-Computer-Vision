from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense, Conv2D
import numpy as np
import cv2
import sys
import gradio as gr
import skimage.segmentation


def find_WBC(image):
    #1
    #segments the image into superpixels
    segments = skimage.segmentation.slic(image, n_segments=100, sigma=5, start_label=0)
    #grabs the count of the groups found
    cnt = len(np.unique(segments))
    #2
    #numpy array to hold the mean color per super pixel
    group_means = np.zeros((cnt, 3), dtype="float32")
    
    #loops through each superpixel index
    for specific_group in range(cnt):
        #creates mask image
        mask_image = np.where(segments == specific_group, 255, 0).astype("uint8")
        #adds channel dimension back to mask image
        mask_image = np.expand_dims(mask_image, axis=-1)
        #computes mean value per group
        group_means[specific_group] = cv2.mean(image, mask=mask_image)[0:3]
    
    #3
    #clusters
    K = 4 
    #criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    #uses k means on the group means to get them into color groups
    retvals, bestLabels, centers = cv2.kmeans(group_means, K, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
    #4
    #hardcoded colors in np arrays
    blue = np.array([255,0,0])
    white = np.array([255,255,255])
    black = np.array([0,0,0])
    #initialize distances list
    distances = []
    #loops through centers
    for center in centers: 
        #gets the closest color to blue
        distance = np.linalg.norm(center - blue)
        #appends closest color to blue to distances
        distances.append(distance)
    #5
    #gets the minimum closest value
    closestColor = np.argmin(distances)
    #iterates throguh the length of centers
    for i in range(len(centers)):
        #if its close, assign it white
        if i == closestColor:
            centers[i] = white
        #else assign it black
        else:
            centers[i] = black
    #6
    #make centers unit8 type
    centers = centers.astype(np.uint8)  
    #gets new superpixel colors
    colors_per_clump = centers[bestLabels.flatten()]
    #7
    #creates a new image
    cell_mask = colors_per_clump[segments]
    #converts to grey
    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
    #8
    #gets disjoint blobs from cell mask
    retval, labels = cv2.connectedComponents(cell_mask, None, 8, cv2.CV_32S)    
    #9
    #initializezs bounding boxes list
    boundingBoxes = []
    #skips 0, loops through retvals (for each blob group)
    for i in range(1, retval):
        #grabs coords of pixels
        coords = np.where(labels == i)
        #if coords exist
        if len(coords[0]) > 0:
            #grabs coordinate values, makes a bounding box and appends it to list of bounding boxes
            #ima be honest idk why i need the axis thing, all i know is it doesn't work with no axis and axis=0
            ymin, xmin = np.min(coords, axis=1)
            ymax, xmax = np.max(coords, axis=1)
            boundingBox = [ymin, xmin, ymax, xmax]
            boundingBoxes.append(boundingBox)
    #returns bounding boxes
    return boundingBoxes
            
