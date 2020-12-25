import numpy as np
import cv2


K = np.load("K.npy")
D = np.load("D.npy")

D=D.squeeze()
print(K)
print(D)
img = cv2.imread('images/left12.jpg')

dst = cv2.undistort(img, K, D, None, None)

'''
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
# undistort
dst = cv2.undistort(img, K, D, None, newcameramtx)
'''

vis = np.concatenate((img, dst), axis=1)
cv2.imwrite('calibresult.png', vis)

