#Upon clicking, saves everything in hdf5 queue. Meant to be click after "Start" has already been pressed


def save(arr_in):
# HDF5 
        try: 
            hdf5.queue.put_nowait((frame_time, data_cube_corr)) 
            num_cubes_stored += 1 # executed if above was successful
        except:
            pass
            # logger.log(logging.WARNING, "HDF5:Storage Queue is full!")

        return()
