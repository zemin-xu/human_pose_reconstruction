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

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

#Estimate the R and T of second view

img1 = cv2.imread("images/left01.jpg")
img2 = cv2.imread("images/left08.jpg")

K = np.load("K.npy")
D = np.load("D.npy")

#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#Find the rotation and translation on view2
ret, corners2 = cv2.findChessboardCorners(img2, (9,6),None)
img2 = cv2.drawChessboardCorners(img2,(9,6),corners2,ret)
ret,rvec2, tvec2 = cv2.solvePnP(objp, corners2, K, D)



#Find the rotation and translation on view1
ret, corners = cv2.findChessboardCorners(img1, (9,6),None)
img1 = cv2.drawChessboardCorners(img1,(9,6),corners,ret)
ret,rvec, tvec = cv2.solvePnP(objp, corners, K, D)




#Form the transformation matrix
P0to1 = homography_transformation(rvec,tvec)
P0to2 = homography_transformation(rvec2,tvec2)
P1to2= np.dot(P0to2,np.linalg.inv(P0to1))
R12 = P1to2[:3,:3]
t12 = P1to2[:3,3:4]
R01,_ = cv2.Rodrigues(rvec)



ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


#The problem starts here
P2 = np.zeros((3,4))
P2[:,:3]= R12
P2[:,3:4]= t12
P2 = np.dot(K,P2)
P1 = np.zeros((3,4))
P1[:,:3]= np.eye(3)
P1 = np.dot(K,P1)
print(P1)
print(P2)
corners = np.squeeze(corners).transpose()
corners2 = np.squeeze(corners2).transpose()
X = cv2.triangulatePoints(P1, P2, corners, corners2)
X=X[:3]/X[3] # 3xN
X = X.transpose()

color = np.ones((X.shape[0],3))*255

fn="chess.ply"
write_ply(fn,X,color)



