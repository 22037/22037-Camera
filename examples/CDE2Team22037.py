import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


while True: 
    current_time = time.time()

    ret,img=cap.read()
    # dimensions = img.shape
    # print(dimensions)

    imgR = img[:,:,0]
    imgG = img[:,:,1]
    imgB = img[:,:,2]

    imgR = np.cast['uint32'](imgR)
    imgR = rebin(imgR, (20,20))
    imgG = np.cast['uint32'](imgG)
    imgG = rebin(imgG, (20,20))
    imgB = np.cast['uint32'](imgB)
    imgB = rebin(imgB, (20,20))
    imgQuant = imgG/imgB

    #imgR = cv2.resize(imgR, [200,200])
    cv2.imshow('Red Channel', imgR)
    #imgG = cv2.resize(imgG, [200,200])
    cv2.imshow('Green Channel', imgG)
    #imgB = cv2.resize(imgB, [200,200])
    cv2.imshow('Blue Channel', imgB)
    #imgQuant = cv2.resize(imgQuant, [200,200])
    cv2.imshow('Division Channel', imgB)
    
    if(cv2.waitKey(10) & 0xFF == ord('q')):
        break


cv2.imwrite("Red Channel.tiff",imgR)
cv2.imwrite("Green Channel.tiff",imgG)
cv2.imwrite("Blue Channel.tiff",imgB)
cv2.imwrite("Division Channel.tiff",imgQuant)
print ("Images written")