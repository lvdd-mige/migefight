# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2

class Crack_Dataset_Test(Dataset):
    def __init__(self, Image, Label):
        self.Image = Image
        self.Label = Label
        # self.Class_Label=np.zeros([1,1])

    def __len__(self):
        return len(self.Image)

    def __getitem__(self, idx):
        # print(self.Image[idx].split("\\")[-1].split(".")[0],self.Label[idx].split("\\")[-1].split(".")[0])
        assert self.Image[idx].split("\\")[-1].split(".")[0] == self.Label[idx].split("\\")[-1].split(".")[0]
        name = self.Image[idx].split("\\")[-1].split(".")[0]
        # cv2.imwrite("Test.png",cv2.imread(self.Label[idx]))
        # cv2.imwrite("Test2.png", cv2.imread(self.Image[idx]))

        image = cv2.resize(cv2.imread(self.Image[idx]), (224, 224)).transpose(2, 0, 1)
        label = cv2.resize(cv2.imread(self.Label[idx]), (224, 224)).transpose(2, 0, 1)

        Class_Label=np.empty([1,])
        if self.Image[idx].split("//")[-1].split("-")[0]=="No":
            Class_Label[0]=0.
        else:
            Class_Label[0]=1.

        label[label < 200] = 0
        label[label > 200] = 1

        # cv2.imwrite("Test1.png",label.transpose(1,2,0))

        image = image.astype(float)
        label = label.astype(float)
        Class_Label=Class_Label.astype(float)

        return image, label,Class_Label, name



class Crack_Dataset(Dataset):
    def __init__(self, Image, Label):
        self.Image = Image
        self.Label = Label
        # self.Class_Label=np.zeros([1,1])

    def __len__(self):
        return len(self.Image)

    def __getitem__(self, idx):
        # print(self.Image[idx].split("\\")[-1].split(".")[0],self.Label[idx].split("\\")[-1].split(".")[0])
        assert self.Image[idx].split("\\")[-1].split(".")[0] == self.Label[idx].split("\\")[-1].split(".")[0]
        name = self.Image[idx].split("\\")[-1].split(".")[0]
        # cv2.imwrite("Test.png",cv2.imread(self.Label[idx]))
        # cv2.imwrite("Test2.png", cv2.imread(self.Image[idx]))

        image = cv2.resize(cv2.cvtColor(cv2.imread(self.Image[idx]),cv2.COLOR_BGR2GRAY), (224, 224))
        label = cv2.resize(cv2.cvtColor(cv2.imread(self.Label[idx]),cv2.COLOR_BGR2GRAY), (224, 224))
        Class_Label=np.zeros([1])
        if len(np.unique(label))==1:
            Class_Label[0]=0
        else:
            Class_Label[0]=1

        label[label < 200] = 0
        label[label > 200] = 1

        # cv2.imwrite("Test1.png",label.transpose(1,2,0))

        image = image.astype(float)
        label = label.astype(float)
        Class_Label=Class_Label

        return image, label,Class_Label, name


def Save_png_Img(Image, name):
    Image[Image > 0.5] = 255
    Image[Image < 0.5] = 0
    cv2.imwrite("Pred_Image\\" + name + ".png", Image[0][0].cpu().detach().numpy())
    return None

def Save_png_Img_Tmp(Image, name):
    Image[Image > 0.5] = 255
    Image[Image < 0.5] = 0
    cv2.imwrite("Pred_Image\\" + name + ".png", Image)
    return None

def Save_png_Img_Pred(Image, name):
    Image[Image > 0.5] = 255
    Image[Image < 0.5] = 0
    cv2.imwrite("Pred_Image\\Pred_Pred\\" + name + ".png", Image)
    return None

def Save_png_Img_Input(Image, name):
    # Image[Image > 0.5] = 255
    # Image[Image < 0.5] = 0
    cv2.imwrite("Pred_Image\\" + name + ".png", Image[0][0].cpu().detach().numpy())
    return None
