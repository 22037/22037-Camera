##########################################################################
# Testing of display and capture & storage thread combined.
##########################################################################
# Results
# =======
# With Queue: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#  Display Rate: 10
#   Frames captured rate: 524.4
#   Cubes generated rate: 37.4
#   Cubes sent to storage rate: 37.4
#   Frames displayed rate: 9.8
#   Cubes stored rate: 37.4 
#   CPU Usage: 6-7%
#   Disk IO: 195MB/s
#  Display Rate: 30
#   Frames captured rate: 524.4
#   Cubes generated rate: 37.4
#   Cubes sent to storage rate: 37.4
#   Frames displayed rate: 27.4
#   Cubes stored rate: 37.6 
#   CPU Usage: 6-7%
#   Disk IO: 195MB/s
# Without Queue:
#  Display Rate: 10
#   Frames captured rate: 240
#   Cubes generated rate: 2.0
#   Cubes sent to storage rate: 2.0
#   Frames displayed rate: 10
#   Cubes stored rate: 1.8
#   CPU Usage: 5-6%
#   Disk IO: MB/s
##########################################################################
import cv2
import logging
import time
import numpy as np
from datetime import datetime

dps_measure_time = 5.0 # average measurements over 5 secs

data_cube = np.zeros((14,540,720), dtype=np.uint8)
frame = np.zeros((540,720), dtype=np.uint8)

# Camera configuration file
from configs.blackfly_configs  import configs

# Display
if configs['displayfps'] >= configs['fps']:
    display_interval = 0
else:
    display_interval = 1.0/configs['displayfps']

window_name    = 'Camera'
font           = cv2.FONT_HERSHEY_SIMPLEX
textLocation0  = (10,480)
textLocation1  = (10,520)
fontScale      = 1
fontColor      = (255,255,255)
lineType       = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Setting up Storage
from camera.streamer.tiffstorageserver import tiffServer
logger.log(logging.INFO, "Starting Storage Server")
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".tiff"
tiff = tiffServer("C:\\temp\\" + filename)
tiff.start()

# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
logger.log(logging.INFO, "Starting Capture")
camera = blackflyCapture(configs)
while not camera.log.empty():
    (level, msg)=camera.log.get_nowait()
    logger.log(level, msg)
camera.start()

# Initialize Variables
frame_idx = 0               # index to create data cube out of individual frames
num_cubes_sent = 0          # keep track of data cubes sent to storage
num_cubes_generated = 0     # keep track of data cubes generated
last_time     = time.time() # keep track of time to dispay performance
last_display  = time.time() # keeo track of time to display images
num_frames_received    = 0  # keep track of how many captured frames reach the main program
num_frames_displayed   = 0  # keep trakc of how many frames are displayed
measured_dps           = 0  # computed in main thread, number of frames displayed per second

# Main Loop
stop = False
while(not stop):
    current_time = time.time()

    # wait for new image
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    data_cube[frame_idx,:,:] = frame
    frame_idx += 1

    while not camera.log.empty():
        (level, msg)=camera.log.get_nowait()
        logger.log(level, msg)

    # When we have a complete dataset:
    if frame_idx >= 14: # 0...13 is populated
        frame_idx = 0
        num_cubes_generated += 1

        if not tiff.queue.full():
            tiff.queue.put_nowait((frame_time, data_cube)) 
            num_cubes_sent += 1
        else:
            logger.log(logging.WARNING, "TIFF:Storage Queue is full!")

    while not tiff.log.empty():
        (level, msg)=tiff.log.get_nowait()
        logger.log(level, msg)

    # Display performance in main loop
    if current_time - last_time >= dps_measure_time:
        # how many data cubes did we create
        measured_cps_generated = num_cubes_generated/dps_measure_time
        logger.log(logging.DEBUG, "Status:captured cubes generated per second:{}".format(measured_cps_generated))
        num_cubes_generated = 0
        # how many data cubes did we send to storage
        measured_cps_sent = num_cubes_sent/dps_measure_time
        logger.log(logging.DEBUG, "Status:cubes sent to storage per second:{}".format(measured_cps_sent))
        num_cubes_sent = 0
        # how many frames did we display
        measured_dps = num_frames_displayed/dps_measure_time
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_time = current_time

    if (current_time - last_display) >= display_interval:
        display_frame = frame.copy()
        # This section creates significant delay and we need to throttle the display to maintain max capture and storage rate
        cv2.putText(display_frame,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(display_frame,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, display_frame)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'): stop = True# this likely is the reason that display frame rate is not faster than 60fps.
        last_display = current_time
        num_frames_displayed += 1

# Cleanup
camera.stop()
tiff.stop()
cv2.destroyAllWindows()
