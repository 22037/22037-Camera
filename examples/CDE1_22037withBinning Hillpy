## Skeleton Code Team 22037
##########################################################################
# Testing of capture and storage thread combined.
# A data cube is 5.5MBytes in size
# It is either copied to shared memory or send via aueue to thread.
# No display of images
# Copyright Dr. Urs Utzinger
##########################################################################

import h5py
import cv2
import logging
import time
import numpy as np
import platform
import os
from datetime import datetime
from timeit import default_timer as timer
from queue import Queue
from configs.blackfly_configs  import configs
from   numba import vectorize, jit, prange
import sys
 
if configs['displayfps'] >= configs['fps']:
    display_interval = 0
else:
    display_interval = 1.0/configs['displayfps']
 
dps_measure_time = 5.0 # average measurements over 5 secs

#configs
res = configs['camera_res']
height = res[1]
width = res[0]
measure_time = 5.0 # average measurements over 5 secs
camera_index = 0 # default camera starts at 0 by operating system

#Processing
looptime = 0.0
use_queue = True
data_cube = np.zeros((14, 540, 720), dtype=np.uint8)
background = np.zeros((540, 720), dtype=np.uint8)
flatfield = np.cast['uint16'](2**8.*np.random.random((540,720)))
data_cube_corr = np.zeros((14, 540, 720), 'uint16')
frame = np.zeros((540,720), dtype=np.uint8)
 
# Camera configuration file
#from configs.blackfly_configs  import configs
 
# Display
window_name    = 'Camera'
font           = cv2.FONT_HERSHEY_SIMPLEX
textLocation0  = (10,480)
textLocation1  = (10,520)
fontScale      = 1
fontColor      = (0,0,255)
lineType       = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
# Display
main_window_name         = 'Captured'
binned_window_name       = 'Binned'
processed_window_name    = 'Band-Passed'
ratioed_window_name      = 'Ratioed'
 
# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

 
# Setting up Storage
from camera.streamer.h5storageserver import h5Server
print("Starting Storage Server")
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
hdf5 = h5Server("C:\\temp\\" + filename)
print("Starting Storage Server")
hdf5.start()
 
# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
print("Starting Capture")
camera = blackflyCapture(configs)
print("Getting Images")
camera.start()
 
# Initialize Variables
frame_idx              = 0  # index to create data cube out of individual frames
num_cubes_stored       = 0  # keep track of data cubes sent to storage
num_cubes_generated    = 0  # keep track of data cubes generated
last_time              = time.perf_counter() # keep track of time to dispay performance
last_display           = time.perf_counter() # keeo track of time to display images
num_frames_received    = 0  # keep track of how many captured frames reach the main program
num_frames_displayed   = 0  # keep trakc of how many frames are displayed
measured_dps           = 0  # computed in main thread, number of frames displayed per second
proc_time              = 0 
counter      = bin_time  = 0  
min_fr = 0.0
max_fr = 1.0
 
 # Reducing the image resolution by binning (summing up pixels)
bin_x=20
bin_y=20
scale = (bin_x*bin_y*255)

# Binning 20 pixels of the 8bit images
@jit(nopython=True, fastmath=True, parallel=True)
def bin20(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//20,n,o), dtype='uint16')
    arr_out = np.empty((m//20,n//20,o), dtype='uint32')
    for i in prange(m//20):
        arr_tmp[i,:,:] =  arr_in[i*20,:,:]    + arr_in[i*20+1,:,:]  + arr_in[i*20+2,:,:]  + arr_in[i*20+3,:,:]  + arr_in[i*20+4,:,:]  + arr_in[i*20+5,:,:]  + \
                          arr_in[i*20+6,:,:]  + arr_in[i*20+7,:,:]  + arr_in[i*20+8,:,:]  + arr_in[i*20+9,:,:]  + arr_in[i*20+10,:,:] + arr_in[i*20+11,:,:] + \
                          arr_in[i*20+12,:,:] + arr_in[i*20+13,:,:] + arr_in[i*20+14,:,:] + arr_in[i*20+15,:,:] + arr_in[i*20+16,:,:] + arr_in[i*20+17,:,:] + \
                          arr_in[i*20+18,:,:] + arr_in[i*20+19,:,:]

    for j in prange(n//20):
        arr_out[:,j,:]  = arr_tmp[:,j*20,:]    + arr_tmp[:,j*20+1,:]  + arr_tmp[:,j*20+2,:]  + arr_tmp[:,j*20+3,:]  + arr_tmp[:,j*10+4,:]  + arr_tmp[:,j*20+5,:]  + \
                          arr_tmp[:,j*20+6,:]  + arr_tmp[:,j*20+7,:]  + arr_tmp[:,j*20+8,:]  + arr_tmp[:,j*20+9,:]  + arr_tmp[:,j*20+10,:] + arr_tmp[:,j*20+11,:] + \
                          arr_tmp[:,j*20+12,:] + arr_tmp[:,j*20+13,:] + arr_tmp[:,j*10+14,:] + arr_tmp[:,j*20+15,:] + arr_tmp[:,j*20+16,:] + arr_tmp[:,j*20+17,:] + \
                          arr_tmp[:,j*20+18,:] + arr_tmp[:,j*20+19,:] 
    return arr_out

@vectorize(['uint16(uint8, uint16, uint8)'], nopython = True, fastmath = True)
def correction(background, flatfield, data_cube):
    return np.multiply(np.subtract(data_cube,background),flatfield)

# Main Loop
stop = False
while(not stop):
    current_time = time.time()
 
    # wait for new image
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    data_cube[frame_idx,:,:] = frame
    num_frames_received += 1
 
    data_cube_corr = correction(background,flatfield, data_cube)
    data_cube_corr[frame_idx,:,:] = frame
    frame_idx += 1
    while not camera.log.empty():
        (level, msg)=camera.log.get_nowait()
        logger.log(level, msg)

    # When we have a complete dataset:
    if frame_idx >= 14: # 0...13 is populated
        frame_idx = 0
        num_cubes_generated += 1

        # HDF5 
        try: 
            hdf5.queue.put_nowait((frame_time, data_cube_corr)) 
            num_cubes_stored += 1 # executed if above was successful
        except:
            pass
            # logger.log(logging.WARNING, "HDF5:Storage Queue is full!")
 
    # Display performance in main loop
# Display performance in main loop
    if current_time - last_time >= measure_time:
        # how much time did it take to process the data
        if num_cubes_generated > 0:
            logger.log(logging.INFO, "Status:process time:{:.2f}ms".format(proc_time*1000./num_cubes_generated))
        # how many data cubes did we create
        measured_cps_generated = num_cubes_generated/measure_time
        logger.log(logging.INFO, "Status:captured cubes generated per second:{}".format(measured_cps_generated))
        num_cubes_generated = 0
        # how many data cubes did we send to storage
        measured_cps_stored = num_cubes_stored/measure_time
        logger.log(logging.INFO, "Status:cubes sent to storage per second:{}".format(measured_cps_stored))
        num_cubes_stored = 0
        # how many frames did we display
        measured_dps = num_frames_displayed/measure_time
        logger.log(logging.INFO, "Status:frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_time = current_time
 
    if (current_time - last_display) >= display_interval:
        display_frame = np.cast['uint8'](data_cube_corr[13,:,:])
        # This section creates significant delay and we need to throttle the display to maintain max capture and storage rate
        cv2.putText(display_frame,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, 255, lineType)
        cv2.putText(display_frame,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, 255, lineType)
        cv2.imshow(window_name, display_frame)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'): # this likely is the reason that display frame rate is not faster than 60fps.
            stop = True
        last_display = current_time
        num_frames_displayed += 1

    #Begin binning for blood quantification
    start_time  = time.perf_counter()
    frame_bin   = bin20(frame)
    bin_time   += (time.perf_counter() - start_time)

    frame_ratio = (frame_bin[:,:,1].astype(np.float32)/frame_bin[:,:,2].astype(np.float32)*255.0).astype(np.uint16)
 
    #Display binned Images
    if (current_time - last_display) >= display_interval:

    #     # Display Camera Data
    #     cv2.putText(frame,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
    #     cv2.imshow(main_window_name, frame)

    #     # Display Binned Image, make it same size as original image
    #     frame_bin_01 = frame_bin/scale # make image 0..1
    #     frame_tmp = cv2.resize(frame_bin_01, (width,height), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    #     cv2.putText(frame_tmp,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
    #     cv2.imshow(binned_window_name, frame_tmp)

    #     # Display Ratio Image, make it same size as original image
    #     frame_ratio_01 = (frame_ratio/255).astype(np.float32)
    #     frame_ratio_01 = np.sqrt(frame_ratio_01)
    #     min_fr = 0.95*min_fr + 0.05*frame_ratio_01.min()
    #     max_fr = 0.95*max_fr + 0.05*frame_ratio_01.max()        
    #     frame_ratio_01 = (frame_ratio_01 -min_fr)/(max_fr-min_fr)
    #     frame_tmp = cv2.resize(frame_ratio_01, (width,height),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    #     cv2.putText(frame_tmp,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
    #     cv2.imshow(ratioed_window_name, frame_tmp)

    #     # # Bandpasse Image, is difference between 0.5Hz and 10Hz
    #     # data_bandpass   = data_lowpass_h - data_lowpass_l

    #     # # transfrom bandpassed image to enhance small changes and to be in 0...1 range
    #     # data_bandpass_tmp = displaytrans(data_bandpass)        
    #     # data_bandpass_tmp = cv2.resize(data_bandpass_tmp, (width,height))
    #     # # Display bandpassed image
    #     # cv2.imshow(processed_window_name, data_bandpass_tmp)

    #     if cv2.waitKey(1) & 0xFF == ord('q'): stop = True

    #     last_display = current_time
    #     counter += 1

    # if (current_time - last_time) >= 5.0: # framearray rate every 5 secs
    #     logger.log(logging.INFO, "Bin:{}".format(bin_time/5.0))
    #     bin_time = 0
    #     last_time = current_time
        print("Images being binned in real time /n")
# Cleanup
camera.stop()
cv2.destroyAllWindows()
