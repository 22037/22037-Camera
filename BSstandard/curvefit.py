######
#curve fit code for senior design main code
#fits curve to each calibration standard image @ each LED

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import image
width  = 720
height = 540
xmin = 0
xmax = width-1
nx =  xmax-xmin+1
ymin = 0
ymax = height-1
ny =  ymax-ymin+1
x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

# Our function to fit is going to be a sum of two-dimensional polynomials
def poly2(x,y,x0,y0,a0,a1,a2,a3,a4,a5):
    x_c = x - x0
    y_c = y - y0
    return a0+ a1*(x_c) + a2*y_c + a3*x_c**2 + a4 * x_c * y_c + a5 * y_c**2
def _poly2(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr = poly2(x, y, *args[0:8])
    return arr

#load in the 14 images based on channel #
C0 = plt.imread("C0-365.tiff")
C1 = plt.imread("C1-460.tiff")
C2 = plt.imread("C2-525.tiff")
C3 = plt.imread("C3-590.tiff")
C4 = plt.imread("C4-623.tiff")
C5 = plt.imread("C5-660.tiff")
C6 = plt.imread("C6-740.tiff")
C7 = plt.imread("C7-850.tiff")
C8 = plt.imread("C8-950.tiff")
C9 = plt.imread("C9-1050.tiff")
C10 = plt.imread("C10-WHITE.tiff")
#C11 = plt.imread("C11.tiff")
C12 = plt.imread("C12-420.tiff")
C13 = plt.imread("C13-BKGND.tiff")

# Initial guesses to the fit parameters
poly2_prms = [720//2, 540//2, 720//2, 540//2, 720//2,  540//2,  720//2,  540//2]

#########################
# Fitting Polynomials
#########################
# 2nd Order
p0 = poly2_prms

# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
xdata = np.vstack((X.ravel(), Y.ravel()))
y_C0 = C0.ravel()
y_C1 = C1.ravel()
y_C2 = C2.ravel()
y_C3 = C3.ravel()
y_C4 = C4.ravel()
y_C5 = C5.ravel()
y_C6 = C6.ravel()
y_C7 = C7.ravel()
y_C8 = C8.ravel()
y_C9 = C9.ravel()
y_C10 = C10.ravel()
#y_C11 = C11.ravel()
y_C12 = C12.ravel()
#y_C13 = C13.ravel()

# Do the fit, using our custom _poly2 function which understands our
# flattened (ravelled) ordering of the data points.
popt0, pcov0 = curve_fit(_poly2, xdata, y_C0, p0)
popt1, pcov1 = curve_fit(_poly2, xdata, y_C1, p0)
popt2, pcov2 = curve_fit(_poly2, xdata, y_C2, p0)
popt3, pcov3 = curve_fit(_poly2, xdata, y_C3, p0)
popt4, pcov4 = curve_fit(_poly2, xdata, y_C4, p0)
popt5, pcov5 = curve_fit(_poly2, xdata, y_C5, p0)
popt6, pcov6 = curve_fit(_poly2, xdata, y_C6, p0)
popt7, pcov7 = curve_fit(_poly2, xdata, y_C7, p0)
popt8, pcov8 = curve_fit(_poly2, xdata, y_C8, p0)
popt9, pcov9 = curve_fit(_poly2, xdata, y_C9, p0)
popt10, pcov10 = curve_fit(_poly2, xdata, y_C10, p0)
#popt11, pcov11 = curve_fit(_poly2, xdata, y_C11, p0)
popt12, pcov12 = curve_fit(_poly2, xdata, y_C12, p0)
#popt13, pcov13 = curve_fit(_poly2, xdata, y_C13, p0)

fit0 = poly2(X, Y, *popt0[0:8])
fit1 = poly2(X, Y, *popt1[0:8])
fit2 = poly2(X, Y, *popt2[0:8])
fit3 = poly2(X, Y, *popt3[0:8])
fit4 = poly2(X, Y, *popt4[0:8])
fit5 = poly2(X, Y, *popt5[0:8])
fit6 = poly2(X, Y, *popt6[0:8])
fit7 = poly2(X, Y, *popt7[0:8])
fit8 = poly2(X, Y, *popt8[0:8])
fit9 = poly2(X, Y, *popt9[0:8])
fit10 = poly2(X, Y, *popt10[0:8])
#fit11 = poly2(X, Y, *popt11[0:8])
fit12 = poly2(X, Y, *popt12[0:8])
#fit13 = poly2(X, Y, *popt13[0:8])

error0 = C0-fit0
error1 = C1-fit1
error2 = C2-fit2
error3 = C3-fit3
error4 = C4-fit4
error5 = C5-fit5
error6 = C6-fit6
error7 = C7-fit7
error8 = C8-fit8
error9 = C9-fit9
error10 = C10-fit10
#error11 = C11-fit11
error12 = C12-fit12
#error13 = C13-fit13

print(error0)