#Calculates and displays synthetic image. 
#Can only be clicked after "start", and "bin" are already active

@jit(nopython=True, fastmath=True, parallel=True)
def blood_psio(arr_in,width,height,scale, counter, textLocation0, font, fontScale, fontColor, lineType):

        frame_ratio = np.divide((arr_in[1,:,:].astype(np.float32),arr_in[2,:,:].astype(np.float32)*255.0).astype(np.uint16))

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