import numpy as np
import cv2

img1 = cv2.imread('images/view0.png',0)  #queryimage # left image
img2 = cv2.imread('images/view1.png',0) #trainimage # right image
h,w = img1.shape[:2]

orb = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

des2 = np.float32(des2)
des1 = np.float32(des1)

FLANN_INDEX_KDTREE = 1
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=100)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flann.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.85*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

vis = np.concatenate((img1, img2), axis=1)

for i in range(len(pts1)):
    tmp = vis.copy()
    pt1 = pts1[i]
    pt2 = pts2[i] + np.array((w,0))
    cv2.circle(tmp,tuple(pts1[i]),5,(255,0,0),1)
    cv2.circle(tmp,tuple(pts2[i]+np.array((w,0))),5,(255,0,0),-1)
    cv2.line(tmp,tuple(pt1),tuple(pt2),(255,0,0),2)
    cv2.imshow("Img",tmp)
    k=cv2.waitKey(0)
    if (k==ord('q')):
         break

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

print(pts1.shape)
print("Rank F ", np.linalg.matrix_rank(F))

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

vis = np.concatenate((img5, img3), axis=1)
cv2.imshow("Img",vis)
k=cv2.waitKey(0)
