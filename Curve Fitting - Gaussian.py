import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import image
width  = 720
height = 540
# The two-dimensional domain of the fit.
xmin = 0
xmax = width-1
nx =  xmax-xmin+1
ymin = 0
ymax = height-1
ny =  ymax-ymin+1
x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)
# Our function to fit is going to be a sum of two-dimensional Gaussians
""" def poly1(x,y,x0,y0,a0,a1,a2):
    x_c = x - x0
    y_c = y - y0
    return a0+ a1*(x_c) + a2*y_c
def _poly1(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr = poly1(x, y, *args[0:5])
    return arr
def poly2(x,y,x0,y0,a0,a1,a2,a3,a4,a5):
    x_c = x - x0
    y_c = y - y0
    return a0+ a1*(x_c) + a2*y_c + a3*x_c**2 + a4 * x_c * y_c + a5 * y_c**2
def _poly2(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr = poly2(x, y, *args[0:8])
    return arr
def poly3(x,y,x0,y0,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9):
    x_c = x - x0
    y_c = y - y0
    x_c_2 = x_c^2
    y_c_2 = y_c^2
    return a0+ a1*(x_c) + a2*y_c + a3*x_c_2 + a4 * x_c * y_c + a5 * y_c_2 + a6*x_c^3 + a7 *x_c_2 * y_c + a8* x_c*y_c_2 + a9 * y_c^3
def _poly3(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr = poly3(x, y, *args[0:12])
    return arr
def poly4(x,y,x0,y0,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14):
    x_c = x - x0
    y_c = y - y0
    x_c_2 = x_c^2
    y_c_2 = y_c^2
    x_c_3 = x_c_2*x_c
    y_c_3 = y_c_2*y_c
    x_c_4 = x_c_3*x_c
    y_c_4 = y_c_3*y_c
    return a0+ a1*(x_c) + a2*y_c + a3*x_c_2 + a4 * x_c * y_c + a5 * y_c_2 + a6*x_c_3 + a7 *x_c_2 * y_c + a8* x_c*y_c_2 + a9 * y_c_3 + a10 *x_c_4 + a11 * x_c_3*y_c_2 + a12 * x_c_2*y_c_2 + a13 * x_c*y_c_3 + a14 * y_c_4
def _poly4(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr = poly4(x, y, *args[0:17])
    return arr """
# This is set to fit multiple Gaussians to the image
#
def gaussian(x, y, x0, y0, xalpha, yalpha, A):
    return A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)
# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//5):
       arr += gaussian(x, y, *args[i*5:i*5+5])
    return arr
# The function to be fit is the white image.
# You need to load the image from disk
#change this line depending on what image you want to look at
Z = plt.imread("21722hand3-02172022172814-22.tiff")
#print (Z.shape)
# Initial guesses to the fit parameters.
#Gaussian Guess for 4 Guassian Cuves
gaussian_prms = [(   0,  0, 1, 1, 2),
                 (-1.5,  5, 5, 1, 3),
                 (  -4, -1, 1.5, 1.5, 6),
                 (   4,  1, 1.5, 1.5, 6.5)
                ]
# Polynominal Guesses for 1st to 4th order
#             x0      y0      a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14
""" poly1_prms = [740//2, 580//2, 1,  1,  1,  1]
poly2_prms = [740//2, 580//2, 1,  1,  1,  1,  1,  1]
poly3_prms = [740//2, 580//2, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
poly4_prms = [740//2, 580//2, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,   1,   1,   1,   1] """
#########################
# Fitting Guassian Curves
#########################
# Flatten the initial guess parameter list.
# from above we have 4 gaussians so we have 4 x prms
p0 = [p for prms in gaussian_prms for p in prms]
# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
xdata = np.vstack((X.ravel(), Y.ravel()))
ydata = Z.ravel()
# Do the fit, using our custom _gaussian function which understands our
# flattened (ravelled) ordering of the data points.
popt, pcov = curve_fit(_gaussian, xdata, ydata, p0)
fit = np.zeros(Z.shape)
for i in range(len(popt)//5):
    fit += gaussian(X, Y, *popt[i*5:i*5+5])
print('Fitted parameters:')
print(popt)
rms = np.sqrt(np.mean((Z - fit)**2))
print('RMS residual =', rms)
# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
ax = plt.subplot(projection='3d')
ax.plot_surface(X, Y, fit, cmap='plasma')
cset = ax.contourf(X, Y, Z-fit, zdir='z', offset=-4, cmap='plasma')
ax.set_zlim(-4,np.max(fit))
plt.show()
# Plot the test data as a 2D image and the fit as overlaid contours.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(Z, origin='lower', cmap='plasma',
          extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(X, Y, fit, colors='w')
plt.show()
#########################
# Fitting Polynoms
#########################
# 2nd Order
# Flatten the initial guess parameter list.
# from above we have 4 gaussians so we have 4 x prms
""" p0 = poly2_prms
# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
xdata = np.vstack((X.ravel(), Y.ravel()))
# Do the fit, using our custom _gaussian function which understands our
# flattened (ravelled) ordering of the data points.
popt, pcov = curve_fit(_poly2, xdata, Z.ravel(), p0)
fit = poly2(X, Y, *popt[0:8])
print('Fitted parameters:')
print(popt)
rms = np.sqrt(np.mean((Z - fit)**2))
print('RMS residual =', rms)
# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
ax = plt.subplot(projection='3d')
ax.plot_surface(X, Y, fit, cmap='plasma')
cset = ax.contourf(X, Y, Z-fit, zdir='z', offset=-4, cmap='plasma')
ax.set_zlim(-4,np.max(fit))
plt.show()
# Plot the test data as a 2D image and the fit as overlaid contours.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(Z, origin='lower', cmap='plasma',
          extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(X, Y, fit, colors='w')
plt.show() """
