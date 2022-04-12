#Meant to be ran after image acquistion "start" button has been pressed

# Variables in: Image frame and appropriate constants: (scale, width, height, counter, textLocation0, font, fontScale, fontColor, lineType)
# Variables out: binned Image frame

# At 14 frame index, the image cube will be binned and then displayed

# Binning 20 pixels of the 8bit images

@jit(nopython=True, fastmath=True, parallel=True)
def binning(arr_in, scale, width, height, counter, textLocation0, font, fontScale, fontColor, lineType):

    frame_idx,n,o = arr_in

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

    if frame_idx >= 14: # 0...13 is populated
        num_cubes_generated += 1

        # Begin Blood Quantification
        start_time = time.time()
        frame_bin   = bin20(arr_in)
        # frame_bin   = rebin(frame, bin_x=20, bin_y=20, dtype=np.uint32)
        bin_time   += (time.perf_counter() - start_time)

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

        return(frame_bin)