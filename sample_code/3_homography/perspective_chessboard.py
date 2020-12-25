import numpy as np
import cv2

img1 = cv2.imread("images/left01.jpg")
img2 = cv2.imread("images/left08.jpg")

vis = np.concatenate((img1,img2),axis=1)
cv2.imshow("img_ori",vis)

ret1, corners1 = cv2.findChessboardCorners(img1, (7,6))
ret2, corners2 = cv2.findChessboardCorners(img2, (7,6))

H, _ = cv2.findHomography(corners1, corners2)
print(H)
img1_warp = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))

vis = np.concatenate((img1_warp,img2),axis=1)
cv2.imshow("img_wrap",vis)
cv2.waitKey(0)
