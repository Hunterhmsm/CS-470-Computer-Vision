import torch
from torch import nn 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
#imports

approaches = ["linear attempt", "random stuff attempt"]

#linear attempt class, rewrote/reused code from torchland
class Linear_Attempt(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()        
        self.net_stack = nn.Sequential(
            #first layer
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(),
            #second layer
            nn.Conv2d(32,32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #third layer
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            #fourth layer
            nn.Linear(4096, 32),
            nn.ReLU(),
            #fifth layer
            nn.Linear(32, 16),
            nn.ReLU(),
            #output layer
            nn.Linear(16, class_cnt)
        )
    def forward(self, x):        
        return self.net_stack(x)
    
class Random_Attempt(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()        
        self.net_stack = nn.Sequential(
            #first layer
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(),
            #2nd layer
            nn.Conv2d(32,32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #third layer
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #4th layer
            nn.ConvTranspose2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            #fifth layer
            nn.LazyLinear(64),
            nn.Sigmoid(),
            #6th Layer
            nn.LazyLinear(32),
            nn.ReLU(),
            #output layer
            nn.Linear(32, class_cnt)
        )
    def forward(self, x):        
        return self.net_stack(x)
    
#returns names of approaches in a list
def get_approach_names():
    return approaches

#returns a description of the approach based on the name
def get_approach_description(approach_name):
    if approach_name == "linear attempt":
        return "Apply conv and linear transformations"
    elif approach_name == "bilinear attempt":
        return "I grabbed random layers from the documentation and added them until it worked"

#returns the transformed data
def get_data_transform(approach_name, training):
    data_transform = v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()])
    return data_transform

#returns batch size
#hard coded single value for now, may change later
def get_batch_size(approach_name):
    batch = 64
    return batch

#creates the model based on name
def create_model(approach_name, class_cnt):
    if approach_name == "linear attempt":
        model = Linear_Attempt(class_cnt)
    elif approach_name == "random stuff attempt":
        model = Random_Attempt(class_cnt)
    

    return model
    
        
#trains the model, reused/rewrote code from Torchland
def train_model(approach_name, model, device, train_dataloader, test_dataloader):
        size = len(train_dataloader.dataset)
        lossFunction = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if approach_name == "linear attempt":
            epochs = 7
            for i in range(epochs):
                print(i+1, " out of ", epochs)
                model.train()
            
                for batch, (X,y) in enumerate(train_dataloader):
                    X = X.to(device)
                    y = y.to(device)
                    
                    pred = model(X)
                    loss = lossFunction(pred, y)
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if batch%100 == 0:
                        loss = loss.item()
                        index = (batch+1)*len(X)
                        print(index, "of", size, ": Loss =", loss)
        elif approach_name == "random stuff attempt":
            epochs = 5
            for i in range(epochs):
                print(i+1, " out of ", epochs)
                model.train()
            
                for batch, (X,y) in enumerate(train_dataloader):
                    X = X.to(device)
                    y = y.to(device)
                    
                    pred = model(X)
                    loss = lossFunction(pred, y)
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if batch%100 == 0:
                        loss = loss.item()
                        index = (batch+1)*len(X)
                        print(index, "of", size, ": Loss =", loss)
        return model