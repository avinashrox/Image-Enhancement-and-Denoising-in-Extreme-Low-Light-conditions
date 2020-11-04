import cv2
import numpy as np
import math
frame = cv2.imread(r'C:\Users\user\Desktop\beautiful-new-zealand.jpg')
gray1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
rows,cols=gray1.shape
gray2=gray1+gray1.std()*np.random.random(gray1.shape)
gray2=np.uint8(gray2)
err = np.sum((gray1.astype("float") - gray2.astype("float")) ** 2)
err /= float(gray1.shape[0] * gray1.shape[1])
psnr=10*math.log10(255*255/err)
print (psnr)
cv2.imshow("gray1",gray1)
cv2.imshow("gray2",gray2)
cv2.waitKey(1)

