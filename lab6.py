import cv2 as cv2
import os
import numpy as np
import time
real_path = '/media/phuc/D/4_years/8_sem/phuc/varotnikov/lab6/Lab6/image/'
font = cv2.FONT_HERSHEY_SIMPLEX

template = cv2.imread(real_path+'template.bmp',0)
w, h = template.shape[::-1]
lower_color = np.array([250, 0, 250])
upper_color = np.array([255, 0, 255])
for img_iter in range(1, 45):
    img_rgb = cv2.imread(real_path + str(img_iter) + '.bmp')

    imgBlack = np.zeros((1080,1920,3), np.uint8)

    # Image Processing in OpenCV/Template Matching
    #https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) # convert image to gray color(ranging from 0 to 255)
    res = cv2.matchTemplate(img_gray,template,cv2.TM_SQDIFF_NORMED)
    minval, maxval, minpos, maxpos = cv2.minMaxLoc(res)
    print(minval,maxval)
    threshold = 0.05 
    loc = np.where( res < threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,0,255), 2)

    mask = cv2.inRange(img_rgb, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height_image, __, _ = img_rgb.shape
    print(len(contours))
    if len(contours) == 50: 
        cv2.putText(img_rgb,'Passed',(10,height_image - 20), font, 6,(0, 255, 0),8)
    else:
        cv2.putText(img_rgb,'Rejected',(10,height_image -20), font, 6,(0, 0, 255),8)
    img_rgb = cv2.resize(img_rgb, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('result', img_rgb)
    cv2.waitKey(1500)
cv2.waitKey(0)
cv2.destroyAllWindows()