import numpy as np
import cv2

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


def homography_transformation(rvec,tvec):
    R,_= cv2.Rodrigues(rvec)
    T = np.zeros((4,4))
    T[3,3]=1
    T[:3,:3]=R
    T[:3,3:4]=tvec
    return T

img1 = cv2.imread("images/left01.jpg")
img2 = cv2.imread("images/left08.jpg")

K = np.load("K.npy")
D = np.load("D.npy")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#Draw
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

#Find the rotation and translation on view2
ret, corners2 = cv2.findChessboardCorners(img2, (9,6),None,flags)
img2 = cv2.drawChessboardCorners(img2,(9,6),corners2,ret)
ret,rvec2, tvec2 = cv2.solvePnP(objp, corners2, K, D)
imgpts, _ = cv2.projectPoints(axis, rvec2, tvec2, K, D)
img = draw(img2,corners2,imgpts)
cv2.imshow('img',img)
k = cv2.waitKey(0)

#Find the rotation and translation on view1
ret, corners = cv2.findChessboardCorners(img1, (9,6),None)
img1 = cv2.drawChessboardCorners(img1,(9,6),corners,ret)
ret,rvec, tvec = cv2.solvePnP(objp, corners, K, D)
imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, D)
img = draw(img1,corners,imgpts)
cv2.imshow('img',img)
k = cv2.waitKey(0)

#Form the transformation matrix
P0to1 = homography_transformation(rvec,tvec)
P0to2 = homography_transformation(rvec2,tvec2)
P1to2= np.dot(P0to2,np.linalg.inv(P0to1))
R12 = P1to2[:3,:3]
t12 = P1to2[:3,3:4]
R01,_ = cv2.Rodrigues(rvec)
#R02,_ = cv2.Rodrigues(rvec2)
#R12= np.dot(R02,R01.transpose())
#t12 = np.dot(R02,np.dot(-R01,tvec))+tvec2

print("R12: ", R12)
print("t12: ",t12)
#print("R12*R12^T: ", np.dot(P1to2[:3,:3],P1to2[:3,:3].transpose()))
#Find the normal vector of plane on first view
normal = np.array((0,0,1)).reshape(3,1)
normal1 = np.dot(R01,normal)
print("Vector normal of plane: ", normal1)
#Find the coordinate of one point of the plane on first view
p  =  np.array((2,3,0)).reshape(3,1)
origin1 = np.dot(R01,p) + tvec.copy()
d1 = -np.dot(normal1.transpose(),origin1)[0]
print("d: ", d1)


H12= R12 - np.dot(t12,normal1.transpose())/d1

G12 = np.dot( K,np.dot(H12,np.linalg.inv(K)) )
H12 = H12/H12[2,2]
G12 = G12/G12[2,2]
print("H12 ", H12)
print("G12", G12)

np.save("G.npy",G12)

vis = np.concatenate((img1,img2),axis=1)
cv2.imshow("img_ori",vis)

img1_warp = cv2.warpPerspective(img1, G12, (img1.shape[1], img1.shape[0]))


vis = np.concatenate((img1_warp,img2),axis=1)
cv2.imshow("img_wrap",vis)
img_mixed = np.uint8(0.5*img1_warp.astype(float)+0.5*img2.astype(float))
cv2.imshow("Overlayed",img_mixed)
cv2.waitKey(0)



