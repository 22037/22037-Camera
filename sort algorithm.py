#code to sort algorithm of images

from threading import Thread
from threading import Lock

import logging
import time
#
import numpy as np
from numba import vectorize

data_cube = np.zeros((14, 4, 4), dtype=np.uint8)

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

data_cube[0,:,:] = array
data_cube[1,:,:] = array2
data_cube[2,:,:] = array3
data_cube[3,:,:] = array4
data_cube[4,:,:] = array5
data_cube[5,:,:] = array6
data_cube[6,:,:] = array7
data_cube[7,:,:] = array8
data_cube[8,:,:] = array9
data_cube[9,:,:] = array10
data_cube[10,:,:] = array11
data_cube[11,:,:] = array12
data_cube[12,:,:] = array13
data_cube[13,:,:] = array14

class algorithm(Thread):

    # Initialize the Processor Thread
    def __init__(self,
        res: tuple = (14, 4, 4),
        bg_delta: tuple = (64, 64)):
        # initialize logger 
        self.logger = logging.getLogger("sort algorithm")

        # Threading Locks, Events
        self.stopped = True
        self.framearray_lock = Lock()

        # Initialize Processor
        self.bg_dx = bg_delta[1]
        self.bg_dy = bg_delta[0]
        self.inten = np.zeros(res[0], 'uint16') 
        self.bg    = np.zeros((res[1],res[2]), 'uint8')
        self.data_cube_corr = np.zeros(res, 'uint16') 

        # Init Frame and Thread
        self.measured_cps = 0.0

        Thread.__init__(self)
    
    def stop(self):
        """stop the thread"""
        self.proc.close()
        self.stopped = True

    def start(self, input_queue):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update, args=(data_cube))
        T.daemon = True # run in background
        T.start()

    # After Starting the Thread, this runs continously
    def update(self, input):
        """ run the thread """
        last_cps_time = time.time()
        num_cubes = 0
        while not self.stopped:
            # Processing throughput calculation
            current_time = time.time()
            if (current_time - last_cps_time) >= 5.0: # framearray rate every 5 secs
                self.measured_cps = num_cubes/5.0
                self.logger.log(logging.INFO, "Status:CPS:{}".format(self.measured_cps))
                num_cubes = 0
                last_cps_time = current_time

            data_cube = input                # Find background image
            _ = np.sum(data_cube[:,::self.bg_dx,::self.bg_dy], axis=(1,2), out = self.inten)
            background_indx = np.argmin(self.inten) # search for minimum intensity 
            self.bg = data_cube[background_indx, :, :]

algorithm(data_cube)

idx = 10
index = 14-idx

index_array = np.arange(0, 14)
array_plus_index = index_array + index
ind = array_plus_index%14

res = [0] * len(data_cube)
for val, idx in zip(data_cube, ind):
    res[idx] = val

data_cube = res

print(data_cube)

#array_new = array + index

#np.array 4x4 make own array
#i = ii+5
#j = i%14
#a=b.copy

#make range: 0-13
#add background index to each wavelength
#divide by modulo 13 - %(#%)

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