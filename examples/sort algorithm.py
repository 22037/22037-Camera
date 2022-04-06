#code to sort algorithm of images

from threading import Thread
from threading import Lock

import logging
import time
#
import numpy as np
from numba import vectorize

import matplotlib.pyplot as plt

data_cube = np.zeros((14, 4, 4), dtype=np.uint8)

#create stand-in arrays - all the same except one is all zeros to simulate background
fit0=plt.imread("C0-365.tiff")
fit1=plt.imread("C1-460.tiff")
fit2=plt.imread("C2-525.tiff")
fit3=plt.imread("C3-590.tiff")
fit4=plt.imread("C4-623.tiff")
fit5=plt.imread("C5-660.tiff")
fit6=plt.imread("C6-740.tiff")
fit7=plt.imread("C7-850.tiff")
fit8=plt.imread("C8-950.tiff")
fit9=plt.imread("C9-1050.tiff")
fit10=plt.imread("C10-WHITE.tiff")
fit11=plt.imread("C13-BKGND.tiff")
fit12=plt.imread("C12-420.tiff")
fit13=plt.imread("C12-420.tiff") + 10

data_cube = np.zeros((14, 540, 720), dtype=np.float32)
data_cube[0,:,:] = fit0
data_cube[1,:,:] = fit1
data_cube[2,:,:] = fit2
data_cube[3,:,:] = fit3
data_cube[4,:,:] = fit4
data_cube[5,:,:] = fit5
data_cube[6,:,:] = fit6
data_cube[7,:,:] = fit7
data_cube[8,:,:] = fit8
data_cube[9,:,:] = fit9
data_cube[10,:,:] = fit10
data_cube[11,:,:] = fit11
data_cube[12,:,:] = fit12
data_cube[13,:,:] = fit13

bg_delta: tuple = (64, 64)
bg_dx = bg_delta[1]
bg_dy = bg_delta[0]

inten = np.sum(data_cube[:,::bg_dx,::bg_dy], axis=(1,2))
background_indx = np.argmin(inten) + 1

index_array = np.arange(0, 14)
array_plus_index = index_array + background_indx
ind = array_plus_index%14

data_cube2 = data_cube[ind,:,:]

print(data_cube)

#print(array_new)