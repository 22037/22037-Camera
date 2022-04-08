#code to fit 2nd order polynomial to any image
#incorporate code into GUI

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import image

width  = 720
height = 540
xmin = 0
xmax = width - 1
nx =  xmax - xmin + 1
ymin = 0
ymax = height - 1
ny =  ymax - ymin + 1
x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

# Our function to fit is going to be a sum of two-dimensional polynomials
def poly2(x, y, x0, y0, a0, a1, a2, a3, a4, a5):
    x_c = x - x0
    y_c = y - y0
    return a0 + a1*(x_c) + a2*y_c + a3*x_c**2 + a4 * x_c * y_c + a5 * y_c**2
def _poly2(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr = poly2(x, y, *args[0:8])
    return arr


######****************** IMAGE ******************######
image = plt.imread("C4-623.tiff")

# define 2nd order fit parameters
p2 = [720//2, 540//2, 720//2, 540//2, 720//2,  540//2,  720//2,  540//2]
# we need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
xdata = np.vstack((X.ravel(), Y.ravel()))
ydata = image.ravel()
# do the fit using our custom poly2 function
popt, pcov = curve_fit(_poly2, xdata, ydata, p2)
# create curve fit
curvefit = poly2(X, Y, *popt[0:8])

######****************** FIT ******************######
# curve fit divided by maximum image value to generate matrix from 0 to 1
fit = curvefit/255.

# OPTIONAL: save fit image as a text file
np.savetxt('fit', fit, delimiter=",")
# code to read text file image: fit = np.loadtxt('fit', dtype='float32', delimiter=',')

Z = image

# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, fit, cmap='plasma')
cset = ax.contourf(X, Y, Z-fit, zdir='z', offset=-4, cmap='plasma')
ax.set_zlim(-4,np.max(fit))
plt.show()