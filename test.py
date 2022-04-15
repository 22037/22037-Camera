from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import h5py
import pyqtgraph as pg

# create some HDF5 data in a 2-d array of X,Y pairs
with h5py.File('plot_2d_data.h5','w') as h5f:
    data = h5f.create_dataset('data',shape=(100,2))
    data[:,0] = np.arange(0.0,10.0,0.1) ## X data points
    data[:,1] = np.random.normal(size=100) ## Y data points

app = QtGui.QApplication([])

win = pg.GraphicsLayoutWidget(show=True, title="2-D plot examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: 2D Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p1 = win.addPlot(title="Plot of NumPy data", 
                 x=np.arange(0.0,10.0,0.1), y=np.random.normal(size=100))

p2 = win.addPlot(title="NumPy data with Points", 
                 x=np.arange(0.0,10.0,0.1), y=np.random.normal(size=100),
                 pen=(255,0,0), symbolBrush=(255,0,0))

win.nextRow()

with h5py.File('plot_2d_data.h5','r') as h5f:
    
    p3 = win.addPlot(title="Plot of HDF5 data", 
                     x=h5f['data'][:,0], y=h5f['data'][:,1])

    p4 = win.addPlot(title="HDF5 data with Points", 
                     x=h5f['data'][:,0], y=h5f['data'][:,1],
                     pen=(0,0,255), symbolBrush=(0,0,255))

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()