#Demo 2 demonstrate the origin shifting from one camera to another 
#using intermediate shifing matrix
#Since origin shifting matrix only available (from 0 to 1), (from 1 to 2),  (from 2 to 3) and (from 3 to 0)
#This example shows the origin shifting of points from (view 0 to  view 2) and then project these points into
#sequence of view 2 

import numpy as np
import cv2
import os
import sys
from tools2D3D import DB2D3DManager


def originShift(point,P):
    point = np.append(point, np.array([1]))
    point = np.dot(P,point)
    return point

dbMan = DB2D3DManager("C:/Users/Zemin XU/PycharmProjects/3d_reconstruction/data")
subject ="Lea"
action ="soulevedeterre_1_0"
nbJ = 15 #Number of joints after triangulation
P01 = dbMan.loadExtrinsic(0) #origin shift matrix from 0 to 1
P12 = dbMan.loadExtrinsic(1) #origin shift matrix from 1 to 2
lst_f,lst_c  = dbMan.loadIntrinsic() #Loading intrinsic parameter of all cameras
pose2Dv2 = dbMan.load2DTXT(subject,action,2) #Load the 2D pose on view 2
pose3Dv0 = dbMan.load3DTXT(subject,action) #Load the 3D pose after triangulation, reference at view 0
cap = dbMan.loadVideoCapture(subject,action,2) #Load the video capture object on view 2
if cap==None:
    exit()
nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
vidWriter = cv2.VideoWriter("demo2.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480)) #VideoWriter for output
assert(pose2Dv2.shape[0]==pose3Dv0.shape[0])
assert(nF == pose2Dv2.shape[0])
step = 255.0/(nbJ-1) 

for i in range(nF):
    sys.stdout.write('\r')
    sys.stdout.write("{}/{}".format(i,nF))
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    ret, frame = cap.read() # Retrieve frame from video capture
    if (not ret):
        exit()
    for j in range(nbJ):
        #Retrieve the X,Y from OpenPose
        X = int(pose2Dv2[i,3*j+0]) 
        Y = int(pose2Dv2[i,3*j+1])

        pos3Dv0 = pose3Dv0[i,4*j+1:4*j+4] #Retrieve the 3D point at view 0 
        pos3Dv1 = originShift(pos3Dv0,P01) #Shifting origin from view 0 to view 1 of 3D point 
        pos3Dv2 = originShift(pos3Dv1,P12) #Shifting origin from view 1 to view 2 of 3D point 
        #Note : you can make a direct shifting matrix from 0 to 2 by transforming 
        #the 3x4 shifting matrix P01,P12 into homogenous matrix 4x4 by adding row at the bottom [0 0 0 1]
        #Then the P02homo = P12homo*P01homo
        pos3Dv2 =pos3Dv2[:2]/pos3Dv2[2] #Normalize 3D coordinate
        pos2Dx = lst_f[2]*pos3Dv2 + lst_c[2] #Project into 2D point on view2, using the intrinsic of cam 2

        cv2.circle(frame,(X,Y),8,(255,255-j*step,j*step),-1) #Draw the points of OpenPose by big color circle
        cv2.circle(frame,tuple(pos2Dx.astype('int')),4,(0,0,255),-1) #Draw the projected point after origin shifting with smaller red circle
        
    vidWriter.write(frame)
print("")
cap.release() #Good pratice
vidWriter.release() #Good pratice
    

