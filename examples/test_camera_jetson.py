import cv2
import logging
import time
import platform
from queue import Queue
import numpy as np

looptime         = 0.0

# Camera configuration file
from configs.nano_IMX219_configs  import configs

if configs['displayfps'] >= configs['fps'] and use_queue:
    display_interval = 0
else:
    display_interval = 1.0/configs['displayfps']
    
window_name      = 'Camera'
font             = cv2.FONT_HERSHEY_SIMPLEX
textLocation0    = (10,20)
textLocation1    = (10,60)
fontScale        = 1
fontColor        = (255,255,255)
lineType         = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("CV2Capture")

# Setting up input and/or output Queue
captureQueue = Queue(maxsize=32)

# Create camera interface
# Based on computer OS you are running
plat = platform.system()
if plat == 'Windows': 
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs)
elif plat == 'Linux':
    if platform.machine() == "aarch64":
        from camera.capture.nanocapture import nanoCapture
        camera = nanoCapture(configs)
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l':
        from camera.capture.cv2capture import cv2Capture
        camera = cv2Capture(configs)
        # from picapture import piCapture
        # camera = piCapture()
elif plat == 'MacOS':
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs)
else:
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs)

print("Getting Images")
camera.start(captureQueue)


# Initialize Variables
last_display  = time.time()
last_fps_time  = time.time()
measured_dps  = 0
num_frames_received    = 0
num_frames_displayed   = 0

while(cv2.getWindowProperty("Camera", 0) >= 0):
    current_time = time.time()
    # wait for new image
    (frame_time, frame) = captureQueue.get(block=True, timeout=None)
    num_frames_received += 1

    if current_time - last_fps_time >= 5.0:
        measured_fps = num_frames_received/5.0
        logger.log(logging.DEBUG, "Status:Frames received per second:{}".format(measured_fps))
        num_frames_received = 0
        measured_dps = num_frames_displayed/5.0
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_fps_time = current_time

    if (current_time - last_display) > display_interval:
        cv2.putText(frame,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, frame)
        # quit the program if users enter q 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        last_display = current_time
        num_frames_displayed += 1

camera.stop()
cv2.destroyAllWindows()
