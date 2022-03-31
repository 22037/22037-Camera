## Main Code Team 22037
##########################################################################
# Testing of capture and storage thread combined.
# Images are displayed & stored

# Copyright Dr. Urs Utzinger
# Edits by sr design Team 22037
##########################################################################

import h5py
import cv2
import logging
import time
import numpy as np
from datetime import datetime
from timeit import default_timer as timer
from queue import Queue
from examples.configs.blackfly_configs  import configs
from numba import vectorize
import matplotlib.pyplot as plt
 
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

#read in curvefit files
#background=np.loadtxt('background', dtype='uint8', delimiter=',')

#images are stored in BSstandard folder
fit0=np.loadtxt('fit0_2', dtype='float32', delimiter=',')
fit1=np.loadtxt('fit1_2', dtype='float32', delimiter=',')
fit2=np.loadtxt('fit2_2', dtype='float32', delimiter=',')
fit3=np.loadtxt('fit3_2', dtype='float32', delimiter=',')
fit4=np.loadtxt('fit4_2', dtype='float32', delimiter=',')
fit5=np.loadtxt('fit5_2', dtype='float32', delimiter=',')
fit6=np.loadtxt('fit6_2', dtype='float32', delimiter=',')
fit7=np.loadtxt('fit7_2', dtype='float32', delimiter=',')
fit8=np.loadtxt('fit8_2', dtype='float32', delimiter=',')
fit9=np.loadtxt('fit9_2', dtype='float32', delimiter=',')
fit10=np.loadtxt('fit10_2', dtype='float32', delimiter=',')
#comment out 11 and 13
fit11=np.loadtxt('fit12_2', dtype='float32', delimiter=',')
fit12=np.loadtxt('fit12_2', dtype='float32', delimiter=',')
fit13=np.loadtxt('background', dtype='float32', delimiter=',')

#Processing
looptime = 0.0
use_queue = True
data_cube = np.zeros((14, 540, 720), dtype=np.uint8)
background = np.zeros((540, 720), dtype=np.uint8)
#background = plt.imread("C13-BKGND.tiff")
flatfield = [fit0, fit1, fit2, fit3, fit4, fit5, fit6, fit7, fit8, fit9, fit10, fit11, fit12, fit13]
#flatfield = [fit0_2, fit1_2, fit2_2, fit3_2, fit4_2, fit5_2, fit6_2, fit7_2, fit8_2, fit9_2, fit10_2, fit11_2, fit12_2, fit13_2]
#flatfield = np.cast['uint8'](2**8.*np.random.random((540,720)))
data_cube_corr = np.zeros((14, 540, 720), 'uint16')
frame = np.zeros((540,720), dtype=np.uint8)

#NEW - FIND BACKGROUND
inten = np.zeros(res[0], 'uint16')
#bg_delta: (int, int) = (64, 64)
bg_delta = np.zeros((64,64), dtype=np.uint64)
bg_dx = bg_delta[1]
bg_dy = bg_delta[0]

#Camera configuration file
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
num_frames_displayed   = 0  # keep track of how many frames are displayed
measured_dps           = 0  # computed in main thread, number of frames displayed per second
proc_time              = 0 
 
@vectorize(['uint16(uint8, float32, uint8)'], nopython = True, fastmath = True)
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

    #NEW - FIND BACKGROUND
    bg_sum = np.sum(data_cube[:,::bg_dx,::bg_dy], axis=(1,2), out = inten)
    background_indx = np.argmin(inten) # search for minimum intensity 
    background = data_cube[background_indx, :, :]

    #b_1 = background_indx+1
    #back1 = data_cube[:b_1]
    #back2 = data_cube[b_1:]
    #data_cube = [*back1, *back2]

    data_cube_corr = correction(background, flatfield, data_cube)
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
 
# Cleanup
camera.stop()
cv2.destroyAllWindows()
