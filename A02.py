import tensorflow
from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense, Conv2D
import numpy as np
import cv2
import sys
import gradio as gr

def main():
    demo = gr.Interface(fn=filtering_callback,
    inputs=["image",
    "file",
    gr.Number(value=0.125),
    gr.Number(value=127)],
    outputs=["image"])
    demo.launch() 

def read_kernel_file(filepath):
    #opens file for reading
    file = open(filepath, 'r')
    #reads first line
    first_line = file.readline()
    #splits using spaces
    tokens = first_line.split(' ')
    #grabs row and col count and converts to int
    rowCnt = int(tokens[0])
    colCnt = int(tokens[1])
    #numpy array of 0s with shape rowCnt and colCnt
    kernel = np.zeros(shape=(rowCnt, colCnt))
    #index starting at 2
    index = 2
    #looping through
    for i in range(rowCnt):
        for j in range(colCnt):
            #converting to float, grabbing the index and storing it. +1 to index
            kernel[i, j] = float(tokens[index])
            index += 1
    #returning kernel
    return kernel

def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uni8=True):
    #casts to float 64
    image = np.array(image, dtype="float64")
    kernel = np.array(kernel, dtype="float64")
    #flips kernel 180 degrees
    kernel = cv2.flip(kernel, -1)
    #integer division to get height and width
    height = kernel.shape[1] // 2
    width = kernel.shape[0] // 2
    #makes padded image hopefully
    padded_image = cv2.copyMakeBorder(image, width, width, height, height, cv2.BORDER_CONSTANT, value=0)
    #output as float hopefully
    output_image = np.zeros(shape=(image.shape), dtype="float64")
    #grabs subimage maybe hopefully
    rows = image.shape[0] 
    cols = image.shape[1]
    # Iterate through the image
    for row in range(rows):
        for col in range(cols):
            subImage = padded_image[ row : (row + kernel.shape[0]), col : (col + kernel.shape[1]) ]
            #filtervals thing
            filtervals = subImage * kernel
            #sum
            value = np.sum(filtervals)
            #writing to output image
            output_image[row,col] = value
    #converts if true
    if convert_uni8 == True:
        output_image = cv2.convertScaleAbs(output_image, alpha=alpha, beta=beta)
    #returns output image
    return output_image


def filtering_callback(input_img, filter_file, alpha_val, beta_val):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    kernel = read_kernel_file(filter_file.name)
    output_img = apply_filter(input_img, kernel, alpha_val, beta_val)
    return output_img

# Later, at the bottom
if __name__ == "__main__":
    main()





            
    

