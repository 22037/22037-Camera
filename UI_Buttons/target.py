# target
#assuming all variables are initialized with start code
#the target code requires variables: self and data_cube

def target(self, data_cube):
    display_interval = 1./300.

    while(not stop):
        stop=self.stop
        current_time = time.time()
        i=(i+1)%14 
        
        # wait for new image
        (self.frame_time, frame) = self.camera.capture.get(block=True, timeout=None)
        data_cube[frame_idx,:,:] = frame
        num_frames_received += 1

        #FIND BACKGROUND         
        if i==13:
            data_cube = sort_algorithm(data_cube)

        data_cube_corr = correction(background, flatfield, data_cube)
        data_cube_corr[frame_idx,:,:] = frame
        frame_idx += 1

        while not self.camera.log.empty():
            (level, msg)=self.camera.log.get_nowait()
            logger.log(level, msg)

        # When we have a complete dataset:
        if frame_idx >= 14: # 0...13 is populated
            frame_idx = 0
            num_cubes_generated += 1       

        if (current_time - last_display) >= display_interval:
                display_frame = np.cast['uint8'](self.data_cube_corr[13,:,:])
                # This section creates significant delay and we need to throttle the display to maintain max capture and storage rate
                cv2.putText(display_frame,"Capture FPS:{} [Hz]".format(self.camera.measured_fps), textLocation0, font, fontScale, 255, lineType)
                cv2.putText(display_frame,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, 255, lineType)
                # cv2.imshow(window_name, display_frame)
                Image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)               
                FlippedImage = cv2.flip(Image1, 1)
                ConvertToQtFormat = QtGui.QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)          
                self.label_CameraDisplay.setPixmap(QPixmap.fromImage(ConvertToQtFormat))                
                self.lcdNumber_FPSin.display(self.camera.measured_fps)
                self.lcdNumber_FPSout.display(measured_dps)

                # quit the program if users enter q or closes the display window
                if cv2.waitKey(1) & 0xFF == ord('q'): # this likely is the reason that display frame rate is not faster than 60fps.
                    stop = True
                last_display = current_time
                num_frames_displayed += 1