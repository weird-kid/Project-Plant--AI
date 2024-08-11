import cv2 as cv
import numpy as np

img=cv.imread('images\jasminepotted.jpg')

hsv_img=cv.cvtColor(img,cv.COLOR_BGR2HSV)

lower_thresh=np.array([36,30,30])
higher_thresh=np.array([70,255,255])

mask=cv.inRange(hsv_img,lower_thresh,higher_thresh)

green=np.zeros_like(img,np.uint8)
green[mask>0]=img[mask>0]

green_map=np.zeros_like(img,np.uint8)
green_map[mask>0]=(0,255,0)
green_pixel=np.count_nonzero(mask)
total_pixel=mask.size
print('number of green pixels',green_pixel)
print('total_pixels:',total_pixel)
cv.addWeighted(img,.7,green_map,0.3
               ,0,green_map)
cv.imshow('greenimage1', green_map)
cv.imshow('greenimage', green)
cv.waitKey()