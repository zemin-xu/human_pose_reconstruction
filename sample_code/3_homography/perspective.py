import numpy as np
import cv2


img_ori = cv2.imread("images/building.jpg")
img = img_ori.copy()
ori_pts = [[280,256],[397,321],[281,309],[400,367]]
dst_pts = [[280,256],[397,256],[280,309],[397,309]]

lst_color=[]
for p in ori_pts:
    color = tuple(np.random.randint(0,255,3).tolist())
    lst_color.append(color)
    cv2.circle(img,tuple(p),5,color,-1)


H, status = cv2.findHomography(np.array(ori_pts),np.array(dst_pts))

print("H: ", H)

img_out = cv2.warpPerspective(img_ori, H, (img_ori.shape[1],img_ori.shape[0]))
for i,p in enumerate(dst_pts):
    cv2.circle(img_out,tuple(p),5,lst_color[i],-1)

vis = np.concatenate((img,img_out),axis=1)
cv2.imshow("img",vis)
cv2.waitKey(0)
