#Team 22037 beginning of blood quantification code

import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

while True: 
    current_time = time.time()

    ret,img = cap.read()
    # dimensions = img.shape
    # print(dimensions)

    
    if(cv2.waitKey(10) & 0xFF == ord('q')):
        break