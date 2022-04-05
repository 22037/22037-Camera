## Binning test

import time
import numpy as np
import platform
import os
from datetime import datetime
from timeit import default_timer as timer
from   numba import vectorize, jit, prange

# Binning 20 pixels of the 8bit images
@jit(nopython=True, fastmath=True, parallel=True)
def bin20(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m,n//20,o), dtype='uint16')
    arr_out = np.empty((m,n//20,o//20), dtype='uint32')
    for i in prange(n//20):
        arr_tmp[:,i,:] =  arr_in[:,i*20,:]    + arr_in[:,i*20+1,:]  + arr_in[:,i*20+2,:]  + arr_in[:,i*20+3,:]  + arr_in[:,i*20+4,:]  + arr_in[:,i*20+5,:]  + \
                          arr_in[:,i*20+6,:]  + arr_in[:,i*20+7,:]  + arr_in[:,i*20+8,:]  + arr_in[:,i*20+9,:]  + arr_in[:,i*20+10,:] + arr_in[:,i*20+11,:] + \
                          arr_in[:,i*20+12,:] + arr_in[:,i*20+13,:] + arr_in[:,i*20+14,:] + arr_in[:,i*20+15,:] + arr_in[:,i*20+16,:] + arr_in[:,i*20+17,:] + \
                          arr_in[:,i*20+18,:] + arr_in[:,i*20+19,:]

    for j in prange(o//20):
        arr_out[:,:,j]  = arr_tmp[:,:,j*20]    + arr_tmp[:,:,j*20+1]  + arr_tmp[:,:,j*20+2]  + arr_tmp[:,:,j*20+3]  + arr_tmp[:,:,j*10+4]  + arr_tmp[:,:,j*20+5]  + \
                          arr_tmp[:,:,j*20+6]  + arr_tmp[:,:,j*20+7]  + arr_tmp[:,:,j*20+8]  + arr_tmp[:,:,j*20+9]  + arr_tmp[:,:,j*20+10] + arr_tmp[:,:,j*20+11] + \
                          arr_tmp[:,:,j*20+12] + arr_tmp[:,:,j*20+13] + arr_tmp[:,:,j*10+14] + arr_tmp[:,:,j*20+15] + arr_tmp[:,:,j*20+16] + arr_tmp[:,:,j*20+17] + \
                          arr_tmp[:,:,j*20+18] + arr_tmp[:,:,j*20+19] 
    return arr_out

data_cube = np.cast['uint16'](np.random.random((14, 540,720)))
print("The shape of a numpy array is: ", np.shape(data_cube))
binned_im = bin20(data_cube)
print("The shape of a numpy array is: ", np.shape(binned_im))
print("done")