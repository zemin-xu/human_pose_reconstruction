import numpy as np
import cv2
import glob
import math
from scipy.optimize import least_squares

def create_vector(H,i,j):
        value = np.zeros((6))
        value[0] = H[0,i]*H[0,j]
        value[1] = H[0,i]*H[1,j]+H[1,i]*H[0,j]
        value[2] = H[1,i]*H[1,j]
        value[3] = H[2,i]*H[0,j]+H[0,i]*H[2,j]
        value[4] = H[2,i]*H[1,j]+H[1,i]*H[2,j]
        value[5] = H[2,i]*H[2,j]
        return value

def create_R(Q):
    u, s, vh = np.linalg.svd(Q)
    return np.dot(u,vh)


def func(x,objpoints,imgpoints):
    Nview = len(objpoints)
    Npoint = objpoints[0].shape[0] 
    ret = np.zeros((2*Nview*Npoint))
    for i in range(Nview):
        img = np.squeeze(imgpoints[i])
        bi = 9+6*i
        R,_ = cv2.Rodrigues(x[bi:bi+3])
        for j in range(Npoint):
            X = R[0,0]*objpoints[i][j,0] \
            +R[0,1]*objpoints[i][j,1] \
            +R[0,2]*objpoints[i][j,2] \
            +x[bi+3]
            Y = R[1,0]*objpoints[i][j,0] \
            +R[1,1]*objpoints[i][j,1] \
            +R[1,2]*objpoints[i][j,2] \
            +x[bi+4]
            Z = R[2,0]*objpoints[i][j,0] \
            +R[2,1]*objpoints[i][j,1] \
            +R[2,2]*objpoints[i][j,2] \
            +x[bi+5]
            X = X/Z 
            Y = Y/Z 
            R2 = X*X+Y*Y
            Xd = X*(1+x[4]*R2+x[5]*R2**2+x[6]*R2**3)+x[7]*(R2+2*X*X)+2*x[8]*X*Y 
            Yd = Y*(1+x[4]*R2+x[5]*R2**2+x[6]*R2**3)+x[8]*(R2+2*Y*Y)+2*x[7]*X*Y
            Xd = x[0]*Xd+x[2]
            Yd = x[1]*Yd+x[3]
            ret[2*i*Npoint+2*j+0]= Xd-img[j,0]
            ret[2*i*Npoint+2*j+1]= Yd-img[j,1]
    return ret

def camera_calib(objpoints,imgpoints):
    Nview = len(objpoints)
    lstH=[]
    V = np.zeros((2*Nview,6))
    for i in range(Nview):
        #Find homography
        H, status = cv2.findHomography(objpoints[i][:,:2], imgpoints[i])
        lstH.append(H)
        V[i*2,:]= create_vector(H,0,1)
        V[2*i+1,:] = create_vector(H,0,0)-create_vector(H,1,1)
    u, s, vh = np.linalg.svd(V, full_matrices=True)
    B = vh[5,:]
    #Find intrinsic parameters
    v0 = (B[1]*B[3]-B[0]*B[4])/(B[0]*B[2]-B[1]*B[1])
    lmda = B[5]-(B[3]*B[3]+v0*(B[1]*B[3]-B[0]*B[4]))/B[0]
    alpha = math.sqrt(lmda/B[0])
    beta = math.sqrt(lmda*B[0]/(B[0]*B[2]-B[1]*B[1]))
    gamma = -B[1]*alpha*alpha*beta/lmda
    u0 = gamma*v0/beta-B[3]*alpha*alpha/lmda

    A = np.zeros((3,3))
    A[0,0]= alpha
    A[0,1] = gamma 
    A[0,2] = u0
    A[1,1] = beta
    A[1,2] = v0 
    A[2,2] = 1

    #Find extinsic parameters
    lstR =[]
    lstt =[]
    invA = np.linalg.inv(A)
    for i in range(Nview):
        R = np.zeros((3,3))
        t = np.zeros((3))
        l = 1 / np.linalg.norm(np.dot(invA,lstH[i][:,0]))
        l2 = 1 / np.linalg.norm(np.dot(invA,lstH[i][:,1]))
        R[:,0] = l*np.dot(invA,lstH[i][:,0])
        R[:,1] = l2*np.dot(invA,lstH[i][:,1])
        R[:,2] = np.cross(R[:,0],R[:,1])
        t = np.dot(invA,lstH[i][:,2])*(l+l2)/2
        R=create_R(R)
        lstR.append(R)
        lstt.append(t)

    x0 = np.zeros(9+Nview*6)
    x0[0] =alpha 
    x0[1] = beta 
    x0[2] = u0
    x0[3] = v0
    for i in range (Nview): 
        bi = 9+6*i
        rvec,_ =cv2.Rodrigues(lstR[i])
        x0[bi:bi+3] = rvec.reshape(3)
        x0[bi+3:bi+6] = lstt[i]

    res_lsq = least_squares(func, x0, args=(objpoints, imgpoints),method='lm')
    print(res_lsq.x[:9])
    print(res_lsq.x[9:16])

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

print(objp)

images = glob.glob("images/*.jpg")

imgpoints = []
objpoints = []
for fname in images:
    img=cv2.imread(fname)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img, (7,6), None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (7,6), corners, ret)
        #cv2.imshow("img",img)
        #cv2.waitKey(0)

camera_calib(objpoints,imgpoints)  

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (640,480), None, None)
print("OpenCV....",mtx)
print(dist)


