import cv2
import numpy as np
from numpy import r_
import scipy
from scipy import misc
from scipy import signal
import math
frame1 = cv2.imread(r'C:\Users\user\Desktop\beautiful-new-zealand.jpg')
gray1=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

imsize = gray1.shape
dct1 = np.zeros(imsize,dtype=np.uint8)

# Do 8x8 DCT on image (in-place)
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct1[i:(i+8),j:(j+8)] = dct2(gray1[i:(i+8),j:(j+8)] )
gray2=gray1+0.25*gray1.std()*np.random.random(gray1.shape)
gray2=np.uint8(gray2)
dct3 = np.zeros(imsize,dtype=np.uint8)

# Do 8x8 DCT on image (in-place)
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct3[i:(i+8),j:(j+8)] = dct2(gray2[i:(i+8),j:(j+8)] )
Tc=np.array([[1.61,2.34,2.57,1.61,1.07,0.64,0.51,0.42],
    [2.14,2.14,1.84,1.35,0.99,0.44,0.43,0.47],
    [1.84,1.98,1.61,1.07,0.64,0.45,0.37,0.46],
    [1.84,1.51,1.17,0.89,0.51,0.29,0.32,0.41],
    [1.43,1.17,0.69,0.46,0.38,0.24,0.25,0.33],
    [1.07,0.73,0.47,0.40,0.32,0.25,0.23,0.28],
    [0.52,0.40,0.33,0.29,0.25,0.22,0.22,0.25],
    [0.38,0.28,0.27,0.26,0.23,0.26,0.25,0.26]])
mse1=0
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        mse1=mse1+np.sum(np.dot((dct1[i:(i+8),j:(j+8)].astype("float")-dct3[i:(i+8),j:(j+8)].astype("float")),Tc.astype("float"))**2)
mse2=mse1/((imsize[0]-7)*(imsize[1]-7)*64)
psnr1=10*math.log10(255*255/mse2)        
print (psnr1)
cv2.imshow('dct1',dct1)
cv2.imshow('dct3',dct3)
cv2.waitKey(0)

