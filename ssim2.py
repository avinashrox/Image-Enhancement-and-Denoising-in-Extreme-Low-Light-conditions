import cv2
import numpy as np
from skimage.measure import compare_ssim
frame= cv2.imread(r'C:\Users\user\Desktop\blue circle1.jpg')
width = int(frame.shape[1] * 0.5)
height = int(frame.shape[0] * 0.25)
dim = (width, height)
frame1= cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
gray1=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
frame2=cv2.imread(r'C:\Users\user\Desktop\white background.jpg')
width = int(frame2.shape[1] * 0.5)
height = int(frame2.shape[0] * 0.25)
dim = (width, height)
frame3= cv2.resize(frame2, dim, interpolation = cv2.INTER_AREA)
gray2=cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY)
(score, diff) = compare_ssim(gray1, gray2, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
cv2.imshow('frame',frame1)
cv2.imshow('gray2',gray2)
cv2.imshow('diff',diff)
cv2.waitKey(1)
