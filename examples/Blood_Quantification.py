#Team 22037 beginning of blood quantification code

import cv2
import numpy as np
import time

# Variables
# Optical constants are given in the units of cm^-1 (dependent on the wavelength)
# h_blood_const =
w_const = 2.79*10**(-3)
# m_const = 
# f_const = 
# mu_a =
# mu_s =
wavelength1 = 540
wavelength2 = 680
# h_blood = 
I_white1 = np.random(720,540)
I_white2 = np.random(720,540)
b1_w1 = 50
b2_w2 = 25
g1 = b1_w1/b2_w2
g1 = b1_w1/b2_w2
R_lambda_1 = I_white1 * ((h_blood)*b1_w1 + b2_w2)
R_lambda_2 = I_white2 * ((h_blood)*b1_w1 + b2_w2)
R = (I_white1/I_white2) * ((h_blood)*g1 + 1)

cap = cv2.VideoCapture(0)

while True: 
    current_time = time.time()

    ret,img = cap.read()

    imgR = img[:,:,0]
    imgG = img[:,:,1]

    # dimensions = img.shape
    # print(dimensions)

    if(cv2.waitKey(10) & 0xFF == ord('q')):
        break