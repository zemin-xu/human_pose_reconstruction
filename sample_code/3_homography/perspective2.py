import numpy as np
import cv2

def cross_norm(vec1,vec2):
    ret = np.cross(vec1,vec2)
    return ret /ret[2]

img_ori = cv2.imread("images/floor.jpg")
img = img_ori.copy()
ori_pts = np.array(((254,497,1),(461,462,1),(277,605,1),(525,552,1)))

dst = [[254,497],[397,497],[254,640],[397,640]]
ori = [[254,497],[461,462],[277,605],[525,552]]

cv2.line(img,tuple(ori_pts[0,:2]),tuple(ori_pts[1,:2]),(0,255,0),5)
cv2.line(img,tuple(ori_pts[2,:2]),tuple(ori_pts[3,:2]),(0,255,0),5)
cv2.line(img,tuple(ori_pts[0,:2]),tuple(ori_pts[2,:2]),(0,0,255),5)
cv2.line(img,tuple(ori_pts[1,:2]),tuple(ori_pts[3,:2]),(0,0,255),5)

l11 = cross_norm(ori_pts[0],ori_pts[1])
l12 = cross_norm(ori_pts[2],ori_pts[3])
pInf1 = cross_norm(l11,l12)

l21 = cross_norm(ori_pts[0],ori_pts[2])
l22 = cross_norm(ori_pts[1],ori_pts[3])
pInf2 = cross_norm(l21,l22)
lInf = cross_norm(pInf1,pInf2)
print(lInf)
print(pInf1,pInf2)
cv2.line(img,tuple(ori_pts[0,:2]),tuple(pInf1[:2].astype(int)),(0,255,255),2)
cv2.line(img,tuple(ori_pts[2,:2]),tuple(pInf1[:2].astype(int)),(0,255,255),2)
cv2.line(img,tuple(ori_pts[2,:2]),tuple(pInf2[:2].astype(int)),(255,100,100),2)
cv2.line(img,tuple(ori_pts[3,:2]),tuple(pInf2[:2].astype(int)),(255,100,100),2)
#H=np.array(((1,0,0.2),(0,1,0.1),(0,0,1)),dtype=float)
#T= np.array(((1,0,0),(0,1,0),(0,0,0)),dtype=float)
#T[2] = lInf
#H = np.dot(H,T)
#print("H", H)
H, status = cv2.findHomography(np.array(ori),np.array(dst))
print("H", H)


img_out = cv2.warpPerspective(img, H, (img_ori.shape[1],img_ori.shape[0]))
vis = np.concatenate((img,img_out),axis=1)
cv2.imshow("img",vis)
cv2.waitKey(0)

