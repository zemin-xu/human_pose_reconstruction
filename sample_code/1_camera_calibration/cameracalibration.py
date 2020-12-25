import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
lst_img = []
images = glob.glob('images/*.jpg')
for fname in images:

    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    #print("A", corners.shape)
    # If found, add object points, image points (after refining them)
    if ret == True:
        lst_img.append(img)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        #print("B", corners2.shape)
        #print(corners2)
        # Draw and display the corners
        #cv.drawChessboardCorners(img, (7,6), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(1)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.save("K.npy",mtx)
np.save("D.npy",dist)
print(mtx)
mean_error = 0
for i in range(len(imgpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )


#Pose Estimation

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img



axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)


for i in range(1):
    img = lst_img[i]
    corners = imgpoints[i]
    rvec = rvecs[i]
    tvec = tvecs[i]
    R,_ = cv.Rodrigues(rvec)
    print(R)
    print(rvec)
    print(tvec)
    imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)    
    img = draw(img,corners,imgpts)
    cv.imshow('img',img)
    k = cv.waitKey(0)





