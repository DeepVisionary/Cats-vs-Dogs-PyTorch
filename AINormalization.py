# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 00:33:27 2023

@author: tommy
"""

import numpy as np
import cv2



x = np.random.rand(3,2)

y = x 

#AdaIN -> Adaptive Instance Normalization, transfer style of y to x channels

new_x = np.std(y)*((x-np.mean(x))/np.std(x)) + np.mean(y)

new_x

img = cv2.imread('C:/Users/tommy/git/Cats-vs-Dogs-PyTorch/Custom Images Pred/basenji.jpg')
img2 = cv2.imread('C:/Users/tommy/git/Cats-vs-Dogs-PyTorch/Custom Images Pred/phoebe.jpg')
import matplotlib.pyplot as plt

plt.imshow(img)

img = np.array(img)

img.shape

def AINorm(img1,img2):
    img1r = img1.ravel()
    img2r = img2.ravel()
    new_x = np.std(img2r)*((img1-np.mean(img1r))/np.std(img1r)) + np.mean(img2r)
    new_x = np.reshape(new_x, img1.shape)
    new_x = np.uint8(new_x)
    return new_x


new_x.shape

plt.imshow(new_x)

new_x = np.uint8(new_x)
plt.imshow(new_x)

img = np.reshape(img, )
img2 = np.reshape(img2, img.shape)
print(img.shape)

img2 = cv2.resize(img2, img.shape)
plt.imshow(img*10)

x = AINorm(img, img2)
plt.imshow(x)
