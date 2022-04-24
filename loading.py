# import sys
# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtGui import QMovie
# from PyQt5.QtCore import Qt


# class LoadingGif(object):

# 	def mainUI(self, FrontWindow):
# 		FrontWindow.setObjectName("FTwindow")
# 		FrontWindow.resize(320, 300)
# 		self.centralwidget = QtWidgets.QWidget(FrontWindow)
# 		self.centralwidget.setObjectName("main-widget")

# 		# Label Create
# 		self.label = QtWidgets.QLabel(self.centralwidget)
# 		self.label.setGeometry(QtCore.QRect(25, 25, 200, 200))
# 		self.label.setMinimumSize(QtCore.QSize(250, 250))
# 		self.label.setMaximumSize(QtCore.QSize(250, 250))
# 		self.label.setObjectName("lb1")
# 		FrontWindow.setCentralWidget(self.centralwidget)

# 		# Loading the GIF
# 		self.movie = QMovie("loader.gif")
# 		self.label.setMovie(self.movie)

# 		self.startAnimation()

# 	# Start Animation

# 	def startAnimation(self):
# 		self.movie.start()

# 	# Stop Animation(According to need)
# 	def stopAnimation(self):
# 		self.movie.stop()


# # app = QtWidgets.QApplication(sys.argv)
# # window = QtWidgets.QMainWindow()
# # demo = LoadingGif()
# # demo.mainUI(window)
# # window.show()
# # sys.exit(app.exec_())


# import numpy as np
# import h5py

# d1 = np.random.random(size = (1000,20))
# d2 = np.random.random(size = (1000,200))
# # print (d1.shape)

# # hf = h5py.File('data.h5', 'w')
# # hf.create_dataset('dataset_1', data=d1)
# # hf.create_dataset('dataset_2', data=d2)
# # # <HDF5 dataset "dataset_2": shape (1000, 200), type "<f8">
# # hf.close()

# hf = h5py.File('20220411133013.hdf5', 'r')
# hf.keys()
# n1 = hf.get('file')

# # n1 = np.array(n1)
# print (hf)

import cv2
import numpy as np
import h5py
# fo = h5py.File('data.h5', 'r')
# pi=fo.keys()
# n1 = fo.get('dataset_1')
f = h5py.File('20220418164453.hdf5', 'r')
p= f.keys()
n1 = f.get('dataset_1')
data = np.array(n1[:,:,:])
file = 'test.jpg'
cv2.imwrite(file, data)

# # import imageio
# import numpy as np
# import h5py

# f = h5py.File('the_file.h5', 'r')
# dset = f['key']
# data = np.array(dset[:,:,:])
# file = 'test.png' # or .jpg
# # imageio.imwrite(file, data)