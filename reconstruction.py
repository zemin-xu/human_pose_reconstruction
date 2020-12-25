import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits import mplot3d
import math
import cv2
import os
import time
import sys
import argparse
from data.tools2D3D import DB2D3DManager

def originShift(point,P):
    point = np.append(point, np.array([1]))
    point = np.dot(P,point)
    return point

##### Global Parameters #####
dbMan = DB2D3DManager("C:/Users/Zemin XU/PycharmProjects/3d_reconstruction/data")
baseDir = "data/"
subject = "Khoa"
action ="burpees_1_0"

##### Task 1: read the 2dTXT file and draw keypoints on original videos #####
txt2dDir = baseDir + "OP2DTXT/" + subject
pose2Dv0 = dbMan.load2DTXT(subject,action,0) #Load the 2D pose data on view 0
pose2Dv0 = np.reshape(pose2Dv0,(pose2Dv0.shape[0],25,3))

# video writer
cap = dbMan.loadVideoCapture(subject,action,0) #Load the video capture object on view 0
nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
renderedName = "OP2DRendered_" + subject + "_" + action + ".avi"
vidWriter = cv2.VideoWriter(renderedName,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480)) #VideoWriter for output
numJoint = 25

for i in range(nFrame):
    # frame by frame operation
    ret, frame = cap.read()
    for j in range(numJoint):
        data2d = pose2Dv0[i][j]
        x = data2d[0].astype('int')
        y = data2d[1].astype('int')
        cv2.circle(frame, (x, y), 4, (255, 255, 255))  # Draw the points of OpenPose by big color circle
    vidWriter.write(frame)


### Task 2: Project these coordinates on the frame of the different camera frame ###
# Read the 3DTXT of any activity, the coordinates 3D has the reference on the first camera (camera 0th).
# Using the relative pose between cameras (pose_0_1.. etc) and camera's intrinsic parameters

pose3Dv0 = dbMan.load3DTXT(subject,action) #Load the 3D pose after triangulation, reference at view 0

P01 = dbMan.loadExtrinsic(0) #origin shift matrix from 0 to 1
P12 = dbMan.loadExtrinsic(1) #origin shift matrix from 1 to 2
lst_f,lst_c  = dbMan.loadIntrinsic() #Loading intrinsic parameter of all cameras

pose2Dv2 = dbMan.load2DTXT(subject,action,2) #Load the 2D pose on view 2
pose3Dv0 = dbMan.load3DTXT(subject,action) #Load the 3D pose after triangulation, reference at view 0

cap = dbMan.loadVideoCapture(subject,action,2) #Load the video capture object on view 2
nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
renderedName = "OP2DCamTransform_" + subject + "_" + action + ".avi"
vidWriter = cv2.VideoWriter(renderedName,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480)) #VideoWriter for output
numJoint = 15

for i in range(nF):
    # frame by frame operation
    ret, frame = cap.read()
    for j in range(numJoint):
        x = pose2Dv2[i, 3*j+0].astype('int')
        y = pose2Dv2[i, 3*j+1].astype('int')

        pos3Dv0 = pose3Dv0[i, 4 * j + 1:4 * j + 4]  # Retrieve the 3D point at view 0
        pos3Dv1 = originShift(pos3Dv0, P01)  # Shifting origin from view 0 to view 1 of 3D point
        pos3Dv2 = originShift(pos3Dv1, P12)  # Shifting origin from view 1 to view 2 of 3D point
        # Note : you can make a direct shifting matrix from 0 to 2 by transforming
        # the 3x4 shifting matrix P01,P12 into homogenous matrix 4x4 by adding row at the bottom [0 0 0 1]
        # Then the P02homo = P12homo*P01homo
        pos3Dv2 = pos3Dv2[:2] / pos3Dv2[2]  # Normalize 3D coordinate
        pos2Dx = lst_f[2] * pos3Dv2 + lst_c[2]  # Project into 2D point on view2, using the intrinsic of cam 2

        cv2.circle(frame, (x, y), 4, (255, 255, 255))
        cv2.circle(frame,tuple(pos2Dx.astype('int')),4,(0,0,255),-1)

    vidWriter.write(frame)

