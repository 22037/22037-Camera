#code to sort algorithm of images

from threading import Thread
from threading import Lock

import logging
import time
#
import numpy as np
from numba import vectorize

data_cube = np.zeros((14, 4, 4), dtype=np.uint8)

#create stand-in arrays - all the same except one is all zeros to simulate background
array = np.arange(0, 16).reshape(4,4)
array2 = np.arange(0, 16).reshape(4,4)
array3 = np.arange(0, 16).reshape(4,4)
array4 = np.arange(0, 16).reshape(4,4)
array5 = np.arange(0, 16).reshape(4,4)
array6 = np.arange(0, 16).reshape(4,4)
array7 = np.arange(0, 16).reshape(4,4)
array8 = np.arange(0, 16).reshape(4,4)
array9 = np.arange(0, 16).reshape(4,4)
array10 = np.zeros((4, 4), dtype='uint8')
array11 = np.arange(0, 16).reshape(4,4)
array12 = np.arange(0, 16).reshape(4,4)
array13 = np.arange(0, 16).reshape(4,4)
array14 = np.arange(0, 16).reshape(4,4)

#this will be the actual data cube in the main code
data_cube[0,:,:] = array
data_cube[1,:,:] = array2+20
data_cube[2,:,:] = array3+40
data_cube[3,:,:] = array4+60
data_cube[4,:,:] = array5+80
data_cube[5,:,:] = array6+100
data_cube[6,:,:] = array7+120
data_cube[7,:,:] = array8+140
data_cube[8,:,:] = array9+160
data_cube[9,:,:] = array10
data_cube[10,:,:] = array11+180
data_cube[11,:,:] = array12+200
data_cube[12,:,:] = array13+220
data_cube[13,:,:] = array14+240

inten = np.sum(data_cube[:,::1,::1], axis=(1,2))
background_indx = np.argmin(inten) + 1

index_array = np.arange(0, 14)
array_plus_index = index_array + background_indx
ind = array_plus_index%14

data_cube = data_cube[ind,:,:]

print(data_cube)

""" bg_delta: tuple = (64, 64)
res: tuple = (14, 4, 4)
bg_dx = bg_delta[1]
bg_dy = bg_delta[0]
inten = np.zeros(res[0], 'uint16') 
bg    = np.zeros((res[1],res[2]), 'uint8')

_ = np.sum(data_cube[:,::bg_dx,::bg_dy], axis=(1,2), out = inten)
background_indx = np.argmin(inten) # search for minimum intensity 
bg = data_cube[background_indx, :, :] """

#print(array_new)