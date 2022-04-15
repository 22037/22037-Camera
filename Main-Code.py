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
from   numba import vectorize, jit, prange
from scipy.optimize import curve_fit

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
flatfield = np.zeros((14, 540, 720), dtype=np.float32)
flatfield[0,:,:] = fit0
flatfield[1,:,:] = fit1
flatfield[2,:,:] = fit2
flatfield[3,:,:] = fit3
flatfield[4,:,:] = fit4
flatfield[5,:,:] = fit5
flatfield[6,:,:] = fit6
flatfield[7,:,:] = fit7
flatfield[8,:,:] = fit8
flatfield[9,:,:] = fit9
flatfield[10,:,:] = fit10
flatfield[11,:,:] = fit11
flatfield[12,:,:] = fit12
flatfield[13,:,:] = fit13
background = np.zeros((540, 720), dtype=np.uint8)
#background = plt.imread("C13-BKGND.tiff")
#flatfield = np.cast['uint8'](2**8.*np.random.random((540,720)))
data_cube_corr = np.zeros((14, 540, 720), 'uint16')
frame = np.zeros((540,720), dtype=np.uint8)

#initialize variables for sort algorithm
bg_delta: tuple = (64, 64)
bg_dx = bg_delta[1]
bg_dy = bg_delta[0]
i=0

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
num_frames_displayed   = 0  # keep track of how many frames are displayed
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

@vectorize(['uint16(uint8, float32, uint8)'], nopython = True, fastmath = True)
def correction(background, flatfield, data_cube):
    return np.multiply(np.subtract(data_cube,background),flatfield)

def sort_algorithm(data):
    inten = np.sum(data[:,::bg_dx,::bg_dy], axis=(1,2))
    background_indx = np.argmin(inten)

    #determine curve fit of background in real time - takes too long
    #flatfield[13,:,:] = wavelength(data_cube[background_indx,:,:])

    index_array = np.arange(0, 14)
    array_plus_index = index_array + background_indx + 1
    ind = array_plus_index%14

    data = data[ind,:,:]

    return data

def wavelength(background):
    width  = 720
    height = 540
    xmin = 0
    xmax = width - 1
    nx =  xmax - xmin + 1
    ymin = 0
    ymax = height - 1
    ny =  ymax - ymin + 1
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # Our function to fit is going to be a sum of two-dimensional polynomials
    def poly2(x, y, x0, y0, a0, a1, a2, a3, a4, a5):
        x_c = x - x0
        y_c = y - y0
        return a0 + a1*(x_c) + a2*y_c + a3*x_c**2 + a4 * x_c * y_c + a5 * y_c**2
    def _poly2(M, *args):
        x, y = M
        arr = np.zeros(x.shape)
        arr = poly2(x, y, *args[0:8])
        return arr
    
    image = background

    # define 2nd order fit parameters
    p2 = [720//2, 540//2, 720//2, 540//2, 720//2,  540//2,  720//2,  540//2]
    # we need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))
    ydata = image.ravel()
    # do the fit using our custom poly2 function
    popt, pcov = curve_fit(_poly2, xdata, ydata, p2)
    # create curve fit
    curvefit = poly2(X, Y, *popt[0:8])

    ######****************** FIT ******************######
    # curve fit divided by maximum image value to generate matrix from 0 to 1
    fit = curvefit/255.

    return(fit)

# Main Loop
stop = False
while(not stop):
    current_time = time.time()
    i=(i+1)%14 
 
    # wait for new image
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    data_cube[frame_idx,:,:] = frame
    num_frames_received += 1

    #NEW - FIND BACKGROUND
    if i==13:
        data_cube = sort_algorithm(data_cube)
        data_cube_corr = correction(background, flatfield, data_cube)


    #data_cube_corr[frame_idx,:,:] = frame
    frame_idx += 1

    while not camera.log.empty():
        (level, msg)=camera.log.get_nowait()
        logger.log(level, msg)

    # When we have a complete dataset:
    if frame_idx >= 14: # 0...13 is populated
        frame_idx = 0
        num_cubes_generated += 1

        """ #Blood Quantification
        start_time = time.time()
        frame_bin   = bin20(data_cube_corr)
        # frame_bin   = rebin(frame, bin_x=20, bin_y=20, dtype=np.uint32)
        bin_time   += (time.perf_counter() - start_time)

        #Green and red
        frame_ratio = (frame_bin[:,:,2].astype(np.float32)/frame_bin[:,:,6].astype(np.float32)*255.0).astype(np.uint16)
        #frame_ratio = (frame_bin[:,:,2].astype(np.float32)/frame_bin[:,:,5].astype(np.float32)*255.0).astype(np.uint16)

        #blue and red
        #frame_ratio = (frame_bin[:,:,1].astype(np.float32)/frame_bin[:,:,6].astype(np.float32)*255.0).astype(np.uint16)
        #frame_ratio = (frame_bin[:,:,1].astype(np.float32)/frame_bin[:,:,5].astype(np.float32)*255.0).astype(np.uint16)

        # Display Binned Image, make it same size as original image
        frame_bin_01 = np.divide(frame_bin,scale) # make image 0..1
        frame_tmp = cv2.resize(frame_bin_01, (width,height), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        cv2.putText(frame_tmp,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
        cv2.imshow(binned_window_name, frame_tmp)

        # Display Ratio Image, make it same size as original image
        frame_ratio_01 = (frame_ratio/255).astype(np.float32)
        frame_ratio_01 = np.sqrt(frame_ratio_01)
        min_fr = np.multiply(0.95,min_fr) + np.multiply(0.05,frame_ratio_01.min())
        max_fr = np.multiply(0.95,max_fr) +np.multiply(0.05,frame_ratio_01.max())       
        frame_ratio_01 = np.divide((np.subtract(frame_ratio_01,min_fr)),(np.subtract(max_fr,min_fr)))
        frame_tmp = cv2.resize(frame_ratio_01, (width,height),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        cv2.putText(frame_tmp,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
        cv2.imshow(ratioed_window_name, frame_tmp) """

        # HDF5 
        try: 
            hdf5.queue.put_nowait((frame_time, data_cube_corr)) 
            # hdf5.queue.put_nowait((frame_time, frame_ratio)) 
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
        display_frame = np.cast['uint8'](data_cube_corr[5,:,:])
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
