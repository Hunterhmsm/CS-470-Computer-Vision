import tensorflow
from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense, Conv2D
import numpy as np
import cv2
import sys
import gradio as gr

#unnormalized hist is just the bin
def create_unnormalized_hist(image):
    #make an array so I can add values later
    unnormalized = np.zeros(shape=(256,), dtype="float32")
    #flatten the image into a 1d array so I can iterate through easy 
    pixels = image.flatten()
    #iterate through the pixels where i is the intensity and update the bins accordingly.
    for i in pixels:
        unnormalized[i] += 1
    return unnormalized

def normalize_hist(hist):
    #calculates the sum
    hist_sum = np.sum(hist)
    #iterates through the array and normalizes it using the sum
    for i in range(len(hist)):
        hist[i] = hist[i] / hist_sum
    return hist

def create_cdf(nhist):
    #np math and fixing type issue thing
    CDF = np.cumsum(nhist.astype('float32'))
    return CDF

    #manual try that doesnt work
    #CDF = np.zeros(shape=(256,), dtype='float32')
    #creates the sum we will add
    #CDF_sum = 0.0
    #iteration = 0
    #should iterate through and add up the sum as it goes.
    #for i in nhist:
        #CDF_sum += i
       #CDF[iteration] = CDF_sum
        #iteration += 1
    #return CDF
    
def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):
    #created unnormalized histogram
    histogram = create_unnormalized_hist(image)
    #creates normalized histogram
    histogram = normalize_hist(histogram)
    #gets the CDF
    CDF = create_cdf(histogram)
    #if stretching is true then does the thing
    #doing the math manually fails the last two tests, swapped to np commands
    if do_stretching == True:
        CDF = np.subtract(CDF, CDF[0])
        CDF = np.divide(CDF, CDF[255])
    #gets transformation and returns it 
    int_transform = CDF * 255.0
    int_transform = cv2.convertScaleAbs(int_transform)[:,0]
    return int_transform

def do_histogram_equalize(image, do_stretching):
    #copies the image
    output = np.copy(image)
    #gets transformation
    transform = get_hist_equalize_transform(image, do_stretching)
    #iterates through the image grabbing the data and transforming then storing it
    for i in range(image.shape[0]): #rows
        for j in range(image.shape[1]): #columns - took way too long to figure this out
            output[i, j] = transform[image[i, j]] #output of (row, col) = transformed image of (row,col)
    return output

    #didnt work, im dumb
    #for i in image:
        #output[i] = transform * i
        
    
def intensity_callback(input_img, do_stretching):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = do_histogram_equalize(input_img, do_stretching)
    return output_img

def main():
    demo = gr.Interface(fn=intensity_callback,
        inputs=["image", "checkbox"],
        outputs=["image"])
    demo.launch()
    
if __name__ == "__main__":
    main()