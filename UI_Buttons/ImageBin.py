#Meant to be ran after image acquistion "start" button has been pressed

# Binning 20 pixels of the 8bit images
@jit(nopython=True, fastmath=True, parallel=True)
def binning(arr_in):

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
        frame_bin   = bin20(data_cube_corr)
        # frame_bin   = rebin(frame, bin_x=20, bin_y=20, dtype=np.uint32)
        bin_time   += (time.perf_counter() - start_time)

        return(frame_bin)