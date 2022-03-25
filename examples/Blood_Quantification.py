#Team 22037 beginning of blood quantification code

import cv2
import numpy as np
import time

# Variables
h_blood_const = # constant values to be looked
w_const = 
m_const =
f_const =
mu_a =
mu_s =
wavelength1 = 540
wavelength2 = 680
h_blood = # blood concentration
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
    # dimensions = img.shape
    # print(dimensions)

    
    

    if(cv2.waitKey(10) & 0xFF == ord('q')):
        break
