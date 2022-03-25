#Team 22037 beginning of blood quantification code

import cv2
import numpy as np
import time

# Variables
# Optical constants are given in the units of cm^-1 (dependent on the wavelength)
h_blood_const =
w_const = 2.79*10**(-3)
m_const = 
f_const = 
mu_a =
mu_s =
wavelength1 = 540
wavelength2 = 680
h_blood = 
wavelength1 = 540
wavelength2 = 680
I_white1 = np.random(720,540)
I_white2 = np.random(720,540)
b1_w1 = 50
b2_w2 = 25
g1 = b1_w1/b2_w2

cap = cv2.VideoCapture(0)

while True: 
    current_time = time.time()

    ret,img = cap.read()
    # dimensions = img.shape
    # print(dimensions)

    
    

    if(cv2.waitKey(10) & 0xFF == ord('q')):
        break