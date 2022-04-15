__author__ = 'Devesh Khosla - github.com/dekhosla'

import sys, serial, serial.tools.list_ports,warnings
from xmlrpc.client import Boolean
from tracemalloc import stop

from PyQt5.QtCore import QSize, QRect,QObject, pyqtSignal, QThread, pyqtSignal, pyqtSlot
import time
from PyQt5.QtWidgets import QApplication, QComboBox,QDialog, QMainWindow, QWidget, QLabel, QTextEdit, QListWidget,QListView

from PyQt5.uic import loadUi
#
import cv2
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import logging
import time
import numpy as np
import cv2
from PyQt5.QtGui import *
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
# from PIL import ImageFont,ImageDraw,Image

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
import matplotlib.pyplot as plt
import sys

# Define Variable

width = 511       # 1920, 720
height = 421      # 1080, 540

display_interval = 1./300.  #
# window_name = 'Camera'

# synthetic data
test_img = np.random.randint(0, 255,
(height, width), 'uint8') # random image

frame = np.zeros((height,width), dtype=np.uint8)
# pre allocate
   

# Setting up logging
logging.basicConfig(level=logging.DEBUG) #options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Display")

font          = cv2.FONT_HERSHEY_SIMPLEX
textLocation0 = (10,20)
textLocation1 = (10,60)
fontScale     = 1
fontColor     = (255,255,255)
lineType      = 2
stop=pyqtSignal(Boolean)


# #defining ports
# ports = [
#             p.device
#             for p in serial.tools.list_ports.comports()
#             if 'USB' in p.description
#         ]
       
# if not ports:
#     raise IOError("There is no device exist on serial port!")
   
# if len(ports) > 1:
#     warnings.warn('Connected....')
# ser = serial.Serial(ports[0],9600)
      
#Port Detection END
# MULTI-THREADING
class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(str)
    @pyqtSlot()

    def __init__(self):
        super(Worker, self).__init__()
        self.working = True

    
    def work(self):
        ser = serial.Serial(self.ports1[0],9600)
        while self.working:
            line =  ser.readline().decode('utf-8')
            print(line)
            time.sleep(0.05)
            self.intReady.emit(line)
        self.finished.emit()

class cameraOnSelected(QObject):
    def on_pushButton_CameraStop_clicked(self):
        z=0
        # self.label_49.clear()   
        # self.camera.stop()
        self.stop=True
        self.stopFuc=False
        ConvertToQtFormat = QtGui.QImage()          
        # self.label_CameraDisplay.setPixmap(QPixmap.fromImage(ConvertToQtFormat))   
        # self.label_CameraDisplay.clear()



class qt(QMainWindow):

    finished = pyqtSignal()
    intReady = pyqtSignal(str)
    @pyqtSlot()
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi('qt.ui', self)
        self.thread = None
        self.worker = None
        # self.find_port()
        # self.stop=pyqtSignal(Boolean)
        self.stop=False
        self.pushButton_StartComm.clicked.connect(self.start_loop)
        # self.label_11.setText("Not detected")
        # self.label_11.setText(ports[0])
        self.menuBar=self.menuBar()               
        self.working=True  
        self.stopFuc=True      
        self.UiComponents()  
        self.onBackground = False
        self.onFlatfield=False
        self.onDatabinning =False
        self.pushButton_CameraOn.clicked.connect(self.on_pushButton_CameraOn)
        self.pushButton_CameraStop.clicked.connect(self.on_pushButton_CameraStop)      
        self.pushButton_CameraSave.clicked.connect(self.on_pushButton_CameraSave) 

        self.pushButton_DefaultView.clicked.connect(self.on_pushButton_DefaultView)      

        self.pushButton_Analysis.clicked.connect(self.on_pushButton_Analysis)      
        self.pushButton_Wavelength.clicked.connect(self.on_pushButton_Wavelength) 
        self.pushButton_Physicogical.clicked.connect(self.on_pushButton_Physicogical)

        self.pushButton_Background.setCheckable(True)
        self.pushButton_Flatfield.setCheckable(True)
        self.pushButton_Databinning.setCheckable(True)

        self.pushButton_Background.clicked.connect(self.on_pushButton_Background)
        self.pushButton_Flatfield.clicked.connect(self.on_pushButton_Flatfield)
        self.pushButton_Databinning.clicked.connect(self.on_pushButton_Databinning)

        self.pushButton_Background.setStyleSheet("background-color : lightgrey")
        self.pushButton_Flatfield.setStyleSheet("background-color : lightgrey")
        self.pushButton_Databinning.setStyleSheet("background-color : lightgrey") 
        
       
     # method for widgets
    def UiComponents(self):

        channel_list = ["All","Channel_1", "Channel_2", "Channel_3", "Channel_4","Channel_5", "Channel_6", "Channel_7", "Channel_8",
        "Channel_9", "Channel_10", "Channel_11", "Channel_12","Channel_13", "Channel_14"]
 
        # adding list of items to combo box
        self.comboBoxDropDown.addItems(channel_list)
        # aboutDisplay="My page"
        # self.textBrowser_About.setText(aboutDisplay)

    def work1(self):
        ser = serial.Serial(self.ports1[0],9600)
        while self.working:
            line =  ser.readline().decode('utf-8')
            print(line)
            time.sleep(0.05)
            self.intReady.emit(line)
        self.finished.emit()

  
    # def portsetup(self):       
    #     #Port Detection START
    #     self.ports = [
    #         p.device
    #         for p in serial.tools.list_ports.comports()
    #         if 'USB' in p.description
    #     ]
       
    #     if self.ports:
    #         if len(self.ports) > 1:
    #             warnings.warn('Connected....')

    #         ser = serial.Serial(self.ports[0],9600)
    #         self.label_11.setText(self.ports[0])

#Port Detection END



    # Save the settings
    def on_pushButton_4_clicked(self):
        if self.x != 0:
            self.textEdit_displayMessage.setText('Settings Saved!')
        else:
            self.textEdit_displayMessage.setText('Please enter port and speed!')

   
  
    # def on_pushButton_clicked(self):
    #     self.completed = 0
    #     while self.completed < 100:
    #         self.completed += 0.001  
    #         self.progressBar.setValue(int(self.completed))
    #     self.textEdit.setText('Data Gathering...')

    #     self.label_5.setText("CONNECTED!")

    #     self.label_5.setStyleSheet('color: green')
    #     x = 1
    #     self.textEdit_3.setText(":")

    def startAnimation(self):
        self.movie.start()
  
    # Stop Animation(According to need)
    def stopAnimation(self):
        self.movie.stop()
  


############################################################### First page Start #############################################
#1.  pushButton_CameraOn_
    def on_pushButton_CameraOn(self):  
        self.stop=False
        self.hdfSave=False
         # Loading the GIF
        self.movie = QMovie("loader.gif")
        self.label_CameraDisplay.setMovie(self.movie)
        self.startAnimation()
        self.label_Status.setText("Status:")
        self.label_SuccesMessage.setText("Started!")
        self.label_SuccesMessage.setStyleSheet('color: blue')
        self.onBackground = False
        self.onFlatfield=False
        self.onDatabinning =False
        self.data_cube_corr = np.zeros((14, 540, 720), 'uint16')
        self.frame = np.zeros((540,720), dtype=np.uint8)
       
        self.on_camera()  

    ############################################################ CAMERA CODE
    def on_camera(self):
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
        self.data_cube_corr = np.zeros((14, 540, 720), 'uint16')
        self.frame = np.zeros((540,720), dtype=np.uint8)
        self.data_cube = np.zeros((14, 540, 720), dtype=np.uint8)
        
        # stop=pyqtSignal(Boolean)

        # Reducing the image resolution by binning (summing up pixels)
        bin_x=20
        bin_y=20
        scale = (bin_x*bin_y*255)
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
        
        # Display
        window_name    = 'Camera'
        font           = cv2.FONT_HERSHEY_SIMPLEX
        textLocation0  = (10,480)
        textLocation1  = (10,520)
        fontScale      = 1
        fontColor      = (0,0,255)
        lineType       = 2

        main_window_name         = 'Captured'
        binned_window_name       = 'Binned'
        processed_window_name    = 'Band-Passed'
        ratioed_window_name      = 'Ratioed'

        # Setting up logging
        logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
        logger = logging.getLogger("Main")
        
        # # Setting up Storage
        from camera.streamer.h5storageserver import h5Server
       
        # Create camera interface
        from camera.capture.blackflycapture import blackflyCapture
        print("Starting Capture")
        self.camera = blackflyCapture(configs)
        print("Getting Images")
        self.camera.start()
                
       # Binning 20 pixels of the 8bit images
        # @jit(nopython=True, fastmath=True, parallel=True)        
        # @vectorize(['uint16(uint8, float32, uint8)'], nopython = True, fastmath = True)
       
        stop=self.stop
        i=0
        while(not stop):
            stop=self.stop
            current_time = time.time()
            i=(i+1)%14 
        
            # wait for new image
            (self.frame_time, frame) = self.camera.capture.get(block=True, timeout=None)
            self.data_cube[frame_idx,:,:] = frame
            num_frames_received += 1

            frame_idx += 1

            while not self.camera.log.empty():
                (level, msg)=self.camera.log.get_nowait()
                logger.log(level, msg)

            # When we have a complete dataset:
            if frame_idx >= 14: # 0...13 is populated
                frame_idx = 0  
                num_cubes_stored = 0

                onFlatfield= self.onFlatfield 
                onBackground = self.onBackground

                # A. Condition for Flat field and Bac 
                if onFlatfield or onBackground:
                    self.data_cube, self.background = self.sort_algorithm()

                    if onFlatfield and onBackground:                         
                         self.data_cube_corr = np.multiply(np.subtract(self.data_cube,self.background),self.flatfield)
                    elif onFlatfield:
                        self.data_cube_corr = np.multiply(self.data_cube,self.flatfield)
                    else:
                       self.data_cube_corr = np.multiply(np.subtract(self.data_cube,self.background))


                # B. Condition for On binning

                onDatabinning= self.onDatabinning
                if onDatabinning:
                    self.data_cube_corr = self.bin20()
                             
                    # HDF5 
                save=self.hdfSave
                if save :
                    frame_idx = 0
                    num_cubes_generated += 1  
                    now = datetime.now() 
                    filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
                    hdf5 = h5Server("C:\\temp\\" + filename)
                    print("Starting Storage Server")
                    hdf5.start()                  
                   
                    try: 
                            # self.hdf5.queue.put_nowait((self.frame_time, self.data_cube_corr)) 
                        hdf5.queue.put_nowait((self.frame_time, self.data_cube_corr)) 
                        num_cubes_stored += 1 # executed if above was successful
                        self.hdfSave=False
                        save=False
                        self.label_Status.setText("Status:")
                        self.label_SuccesMessage.setText("Saved!")
                        self.label_SuccesMessage.setStyleSheet('color: green')
                    except:
                        pass
                            # logger.log(logging.WARNING, "HDF5:Storage Queue is full!")
                    
                    hdf5.stop()  
        
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
                
                display_frame = np.cast['uint8'](self.data_cube_corr[1,:,:])
                # This section creates significant delay and we need to throttle the display to maintain max capture and storage rate
                cv2.putText(display_frame,"Capture FPS:{} [Hz]".format(self.camera.measured_fps), textLocation0, font, fontScale, 255, lineType)
                cv2.putText(display_frame,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, 255, lineType)
                # cv2.imshow(window_name, display_frame)
                Image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)               
                FlippedImage = cv2.flip(Image1, 1)               
                ConvertToQtFormat = QtGui.QImage(FlippedImage.data, FlippedImage.shape[1],FlippedImage.shape[0], QImage.Format_RGB888)          
                self.label_CameraDisplay.setPixmap(QPixmap.fromImage(ConvertToQtFormat))                
                self.lcdNumber_FPSin.display(self.camera.measured_fps)
                self.lcdNumber_FPSout.display(measured_dps)

                # quit the program if users enter q or closes the display window
                if cv2.waitKey(1) & 0xFF == ord('q'): # this likely is the reason that display frame rate is not faster than 60fps.
                    stop = True
                last_display = current_time
                num_frames_displayed += 1
                self.stopAnimation()

     #curve fit function   
    def curveFitFlatField(self):
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
        self.flatfield=flatfield

        self.background = np.zeros((540, 720), dtype=np.uint8)
       
    # sorting function
    def sort_algorithm(self):        
            bg_delta: tuple = (64, 64)
            bg_dx = bg_delta[1]
            bg_dy = bg_delta[0]
            inten = np.sum(self.data_cube[:,::bg_dx,::bg_dy], axis=(1,2))
            background_indx = np.argmin(inten)

            background = self.data_cube[background_indx,:,:]
            index_array = np.arange(0, 14)
            array_plus_index = index_array + background_indx + 1
            ind = array_plus_index%14

            self.data_cube = self.data_cube[ind,:,:]

            return self.data_cube, self.background
            
    def bin20(self):
        arr_in=self.data_cube_corr
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
       
        self.data_cube_corr= arr_out
        return self.data_cube_corr

 
    ############################################################END CAMERA CODE     

#2. Camera Stop spin view Button
    def on_pushButton_CameraStop(self): 
        self.label_Status.setText("Status:")
        self.label_SuccesMessage.setText("Stopped!")
        self.label_SuccesMessage.setStyleSheet('color: red')
        self.startAnimation() 
        self.stop=True
        self.camera.stop()       
        ConvertToQtFormat = QtGui.QImage()          
        self.label_CameraDisplay.setPixmap(QPixmap.fromImage(ConvertToQtFormat))   
        self.label_CameraDisplay.clear()
        self.stopAnimation() 

#3. Camera Save pushButton_CameraSave
    def on_pushButton_CameraSave(self): 
        self.startAnimation() 
        self.hdfSave=True  
  

# 4. Display Target or default view pushButton_DefaultView_
    def on_pushButton_DefaultView(self):     
        z=0
    def target(window_name, display_frame):
      cv2.imshow(window_name, display_frame)

#  5.  Analysis on_pushButton_Analysis_clicked
    def on_pushButton_Analysis(self):        
        z=0
# 6. Okay Button pushButton_Wavelength
    def on_pushButton_Wavelength(self):   
        content=self.comboBoxDropDown.currentText()
        self.wavelengthSelected=content
        z=0
    
# 7.  pushButton_Physicogical
    def on_pushButton_Physicogical(self):   
        self.blood_psio(self)

# 8. check buttons 
    def on_pushButton_Background(self):
        if self.pushButton_Background.isChecked():
            self.pushButton_Background.setStyleSheet("background-color : limegreen")
            self.pushButton_Background.setText("On")
            self.onBackground = True
        else:
            self.pushButton_Background.setStyleSheet("background-color : lightgrey")
            self.pushButton_Background.setText("Off")
            self.onBackground = False 

    def on_pushButton_Flatfield(self):
        if self.pushButton_Flatfield.isChecked():
            self.pushButton_Flatfield.setStyleSheet("background-color : limegreen")
            self.pushButton_Flatfield.setText("On")
            self.curveFitFlatField()
            self.onFlatfield = True
        else:
            self.pushButton_Flatfield.setStyleSheet("background-color : lightgrey")
            self.pushButton_Flatfield.setText("Off")
            self.onFlatfield = False

    def on_pushButton_Databinning(self):
        if self.pushButton_Databinning.isChecked():
            self.pushButton_Databinning.setStyleSheet("background-color : limegreen")
            self.pushButton_Databinning.setText("On")
            self.onDatabinning = True
        else:
            self.pushButton_Databinning.setStyleSheet("background-color : lightgrey")
            self.pushButton_Databinning.setText("Off")
            self.onDatabinning = False

    #  @jit(nopython=True, fastmath=True, parallel=True)
    def blood_psio(self):
        # Reducing the image resolution by binning (summing up pixels)
        bin_x=20
        bin_y=20
        counter      = bin_time  = 0  
        scale = (bin_x*bin_y*255)
        start_time = time.time()
        frame_bin   = self.bin20(self.data_cube_corr)
                # frame_bin   = rebin(frame, bin_x=20, bin_y=20, dtype=np.uint32)
        bin_time   += (time.perf_counter() - start_time)

        frame_ratio = np.divide((frame_bin[1,:,:].astype(np.float32),frame_bin[2,:,:].astype(np.float32)*255.0).astype(np.uint16))

        # Display Binned Image, make it same size as original image
        frame_bin_01 = frame_bin/scale # make image 0..1
        frame_tmp = cv2.resize(frame_bin_01, (width,height), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        cv2.putText(frame_tmp,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
        # cv2.imshow(binned_window_name, frame_tmp)
        Image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)               
        FlippedImage = cv2.flip(Image1, 1)
        ConvertToQtFormat = QtGui.QImage(FlippedImage.data, FlippedImage.shape[1],FlippedImage.shape[0], QImage.Format_RGB888)          
       
        self.label_CameraDisplay.setPixmap(QPixmap.fromImage(ConvertToQtFormat))                
        self.lcdNumber_FPSin.display(self.camera.measured_fps)
        self.lcdNumber_FPSout.display(counter)



        # Display Ratio Image, make it same size as original image
        frame_ratio_01 = (frame_ratio/255).astype(np.float32)
        frame_ratio_01 = np.sqrt(frame_ratio_01)
        min_fr = 0.95*min_fr + 0.05*frame_ratio_01.min()
        max_fr = 0.95*max_fr + 0.05*frame_ratio_01.max()        
        frame_ratio_01 = (frame_ratio_01 -min_fr)/(max_fr-min_fr)
        frame_tmp = cv2.resize(frame_ratio_01, (width,height),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        cv2.putText(frame_tmp,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
        # cv2.imshow(ratioed_window_name, frame_tmp)
        Image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)               
        FlippedImage = cv2.flip(Image1, 1)
        ConvertToQtFormat = QtGui.QImage(FlippedImage.data, FlippedImage.shape[1],FlippedImage.shape[0], QImage.Format_RGB888)    

        self.label_CameraDisplay.setPixmap(QPixmap.fromImage(ConvertToQtFormat))                
        self.lcdNumber_FPSin.display(self.camera.measured_fps)
        self.lcdNumber_FPSout.display(counter)


        return (frame_ratio)

   
############################################################### First page End ###############################################

############################################################### Second page Start ############################################
    def find_port(self):
        self.label_CameraDisplay.clear()

        #defining ports
        self.ports1 = [
            p.device
            for p in serial.tools.list_ports.comports()
            if 'USB' in p.description
        ]
       
        if not self.ports1:
         raise IOError("There is no device exist on serial port!")
   
        if len(self.ports1) > 1:
         warnings.warn('Connected....')

        # self.selectedSerial = serial.Serial(self.ports1[0],9600)
        # self.label_11.setText(self.ports1[0])

    def on_pushButton_StartComm_clicked(self):
      
        self.completed = 0
        while self.completed < 100:
            self.completed += 0.001  
            self.progressBar.setValue(int(self.completed))
        self.textEdit_displayMessage.setText('Data Gathering...')

        self.label_PortStatus.setText("CONNECTED!")

        self.label_PortStatus.setStyleSheet('color: green')
        x = 1
        self.textEdit_displayMessage.setText(":")
    
    def on_pushButton_StopComm_clicked(self):
        self.textEdit_displayMessage.setText('Stopped! Please click CONNECT...')

    
    def on_pushButton_SendComm_clicked(self):
        # Send data from serial port:
        mytext = self.textEdit_TextSendDisplay.toPlainText()
        # self.portsetup(self)
        print(mytext.encode())
        self.selectedSerial.write(mytext.encode())
    
    
    def stop_loop(self):
        self.worker.working = False

    def onIntReady(self, i):
        self.textEdit_DisplayCommData.append("{}".format(i))
        print(i)
    def loop_finished(self):
        print('Loop Finished')

    def start_loop(self):
        # self.portsetup()
        if self.ports1:           
            self.worker = Worker()   # a new worker to perform those tasks
            self.thread = QThread()  # a new thread to run our background tasks in
            self.worker.moveToThread(self.thread)  # move the worker into the thread,do this first before connecting the signals
            self.thread.started.connect(self.work) # begin our worker object's loop when the thread starts running

            self.worker.intReady.connect(self.onIntReady)
            self.pushButton_StopComm.clicked.connect(self.stop_loop)      # stop the loop on the stop button click
            self.worker.finished.connect(self.loop_finished)       #do something in the gui when the worker loop ends

            self.worker.finished.connect(self.thread.quit)    # tell the thread it's time to stop running
            self.worker.finished.connect(self.worker.deleteLater)  #have worker mark itself for deletion
            self.thread.finished.connect(self.thread.deleteLater) # have thread mark itself for deletion

            self.thread.start()
        if not self.ports1:
            self.label_11.setText("Nothing found")
 

 ############################################################### Second page Start ############################################
    def find_port(self):
        self.label_CameraDisplay.clear()

        #defining ports
        self.ports1 = [
            p.device
            for p in serial.tools.list_ports.comports()
            if 'USB' in p.description
        ]
       
        if not self.ports1:
         raise IOError("There is no device exist on serial port!")
   
        if len(self.ports1) > 1:
         warnings.warn('Connected....')

        # self.selectedSerial = serial.Serial(self.ports1[0],9600)
        # self.label_11.setText(self.ports1[0])

    def on_pushButton_StartComm_clicked(self):
      
        self.completed = 0
        while self.completed < 100:
            self.completed += 0.001  
            self.progressBar.setValue(int(self.completed))
        self.textEdit_displayMessage.setText('Data Gathering...')

        self.label_PortStatus.setText("CONNECTED!")

        self.label_PortStatus.setStyleSheet('color: green')
        x = 1
        self.textEdit_displayMessage.setText(":")
    
    def on_pushButton_StopComm_clicked(self):
        self.textEdit_displayMessage.setText('Stopped! Please click CONNECT...')

    
    def on_pushButton_SendComm_clicked(self):
        # Send data from serial port:
        mytext = self.textEdit_TextSendDisplay.toPlainText()
        # self.portsetup(self)
        print(mytext.encode())
        self.selectedSerial.write(mytext.encode())
    
    
    def stop_loop(self):
        self.worker.working = False

    def onIntReady(self, i):
        self.textEdit_DisplayCommData.append("{}".format(i))
        print(i)
    def loop_finished(self):
        print('Loop Finished')

    def start_loop(self):
        # self.portsetup()
        if self.ports1:           
            self.worker = Worker()   # a new worker to perform those tasks
            self.thread = QThread()  # a new thread to run our background tasks in
            self.worker.moveToThread(self.thread)  # move the worker into the thread,do this first before connecting the signals
            self.thread.started.connect(self.work) # begin our worker object's loop when the thread starts running

            self.worker.intReady.connect(self.onIntReady)
            self.pushButton_StopComm.clicked.connect(self.stop_loop)      # stop the loop on the stop button click
            self.worker.finished.connect(self.loop_finished)       #do something in the gui when the worker loop ends

            self.worker.finished.connect(self.thread.quit)    # tell the thread it's time to stop running
            self.worker.finished.connect(self.worker.deleteLater)  #have worker mark itself for deletion
            self.thread.finished.connect(self.thread.deleteLater) # have thread mark itself for deletion

            self.thread.start()
        if not self.ports1:
            self.label_11.setText("Nothing found")
 
 

   

############################################################### Second page Start ############################################
    def find_port(self):
        self.label_CameraDisplay.clear()

        #defining ports
        self.ports1 = [
            p.device
            for p in serial.tools.list_ports.comports()
            if 'USB' in p.description
        ]
       
        if not self.ports1:
         raise IOError("There is no device exist on serial port!")
   
        if len(self.ports1) > 1:
         warnings.warn('Connected....')

        # self.selectedSerial = serial.Serial(self.ports1[0],9600)
        # self.label_11.setText(self.ports1[0])

    def on_pushButton_StartComm_clicked(self):
      
        self.completed = 0
        while self.completed < 100:
            self.completed += 0.001  
            self.progressBar.setValue(int(self.completed))
        self.textEdit_displayMessage.setText('Data Gathering...')

        self.label_PortStatus.setText("CONNECTED!")

        self.label_PortStatus.setStyleSheet('color: green')
        x = 1
        self.textEdit_displayMessage.setText(":")
    
    def on_pushButton_StopComm_clicked(self):
        self.textEdit_displayMessage.setText('Stopped! Please click CONNECT...')

    
    def on_pushButton_SendComm_clicked(self):
        # Send data from serial port:
        mytext = self.textEdit_TextSendDisplay.toPlainText()
        # self.portsetup(self)
        print(mytext.encode())
        self.selectedSerial.write(mytext.encode())
    
    
    def stop_loop(self):
        self.worker.working = False

    def onIntReady(self, i):
        self.textEdit_DisplayCommData.append("{}".format(i))
        print(i)
    def loop_finished(self):
        print('Loop Finished')

    def start_loop(self):
        # self.portsetup()
        if self.ports1:           
            self.worker = Worker()   # a new worker to perform those tasks
            self.thread = QThread()  # a new thread to run our background tasks in
            self.worker.moveToThread(self.thread)  # move the worker into the thread,do this first before connecting the signals
            self.thread.started.connect(self.work) # begin our worker object's loop when the thread starts running

            self.worker.intReady.connect(self.onIntReady)
            self.pushButton_StopComm.clicked.connect(self.stop_loop)      # stop the loop on the stop button click
            self.worker.finished.connect(self.loop_finished)       #do something in the gui when the worker loop ends

            self.worker.finished.connect(self.thread.quit)    # tell the thread it's time to stop running
            self.worker.finished.connect(self.worker.deleteLater)  #have worker mark itself for deletion
            self.thread.finished.connect(self.thread.deleteLater) # have thread mark itself for deletion

            self.thread.start()
        if not self.ports1:
            self.label_11.setText("Nothing found")
 
 

   
############################################################### Second page End ##############################################

############################################################### Third page Start #############################################

############################################################### Third page End ###############################################
def run():
    app = QApplication(sys.argv)
    widget = qt()
    widget.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()
