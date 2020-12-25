import numpy as np 
import cv2 
from scipy.optimize import least_squares

K = np.load("K.npy")
D = np.load ("D.npy")
D = np.squeeze(D)
print("D", D)

def func(x,objpoints,imgpoints,K,D):
    imgpoints = np.squeeze(imgpoints)
    Npoints = objpoints.shape[0]
    R ,_= cv2.Rodrigues(x[:3])
    
    ret = np.zeros((Npoints*2))
    for i in range (Npoints):
        
        X = np.dot(R,objpoints[i]) + x[3:]
        X=X/X[2]
        Xd = X.copy()
        R2 = X[0]**2+X[1]**2
        Xd[0] = X[0]*(1+D[0]*R2+D[1]*R2**2+D[4]*R2**3)+ 2*D[2]*X[0]*X[1] + D[3]*(R2+2*X[0]**2)
        Xd[1] = X[1]*(1+D[0]*R2+D[1]*R2**2+D[4]*R2**3)+ 2*D[3]*X[0]*X[1] + D[2]*(R2+2*X[1]**2)
        Xd = np.dot(K,Xd)
        ret[i*2+0]=imgpoints[i,0]-Xd[0]
        ret[i*2+1]=imgpoints[i,1]-Xd[1]
        '''
        X = R[0,0]*objpoints[i,0] \
            +R[0,1]*objpoints[i,1] \
            +R[0,2]*objpoints[i,2] \
            +x[3]
        Y = R[1,0]*objpoints[i,0] \
            +R[1,1]*objpoints[i,1] \
            +R[1,2]*objpoints[i,2] \
            +x[4]
        Z = R[2,0]*objpoints[i,0] \
            +R[2,1]*objpoints[i,1] \
            +R[2,2]*objpoints[i,2] \
            +x[5]
        X = X/Z 
        Y = Y/Z 
        R2 = X*X+Y*Y
        Xd = X*(1+D[0]*R2+D[1]*R2**2+D[4]*R2**3)+ 2*D[2]*X*Y + D[3]*(R2+2*X*X)
        Yd= Y*(1+D[0]*R2+D[1]*R2**2+D[4]*R2**3)+ 2*D[3]*X*Y + D[2]*(R2+2*Y**2)
        Xd = Xd*K[0,0]+K[0,2]
        Yd = Yd*K[1,1] +K[1,2]
        ret[i*2+0]=imgpoints[i,0]-Xd
        ret[i*2+1]=imgpoints[i,1]-Yd
        '''
    return ret
def findCameraPose(objpoints,imgpoints,K,D):
    x0 = np.zeros((6))
    #x0[0] = -0.4
    #x0[1] = 0.2
    #x0[2] = -2.9
    x0[3] = 0
    x0[4] = 0
    x0[5] = 0.3
    
    #func(x0,objpoints,imgpoints,K,D)
    fakeD = np.zeros(5)
    res_lsq = least_squares(func, x0, args=(objpoints, imgpoints,K,fakeD))
    #x0 = res_lsq.x
    #res_lsq = least_squares(func, x0, args=(objpoints, imgpoints,K,D))
    print(res_lsq.cost)
    return res_lsq.x[:3], res_lsq.x[3:]
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

img = cv2.imread("images/left01.jpg")
    
ret, corners = cv2.findChessboardCorners(img, (7,6),None)
if ret == True:
    #corners2 = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
    # Find the rotation and translation vectors.
    ret,rvecs, tvecs = cv2.solvePnP(objp, corners, K, D)
    R1,_ = cv2.Rodrigues(rvecs)
    print(R1,"\n",tvecs)
    rvec,tvec = findCameraPose(objp,corners,K,D)
    R2,_ = cv2.Rodrigues(rvec)
    print(R2,"\n",tvec)


    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K, D)
    img1 = draw(img.copy(),corners,imgpts)
    cv2.imshow('img',img1)
    print(imgpts)
    imgpts2, jac2 = cv2.projectPoints(axis, rvec, tvec, K, D)
    print(imgpts2)
    img2 = draw(img.copy(),corners,imgpts2)
    cv2.imshow('img2',img2)
    k = cv2.waitKey(0) & 0xFF
#cv2.destroyAllWindows()
