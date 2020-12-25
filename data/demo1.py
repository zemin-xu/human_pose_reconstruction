#Demo 1 demonstrate the origin shifting from one camera to another

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
subject ="Khoa"
action ="soulevedeterre_1_0"
nbJ = 15 #Number of joints after triangulation
P01 = dbMan.loadExtrinsic(0) #origin shift matrix from 0 to 1
lst_f,lst_c  = dbMan.loadIntrinsic() #Loading intrinsic parameter of all cameras
pose2Dv1 = dbMan.load2DTXT(subject,action,1) #Load the 2D pose on view 1
pose3Dv0 = dbMan.load3DTXT(subject,action) #Load the 3D pose after triangulation, reference at view 0
cap = dbMan.loadVideoCapture(subject,action,1) #Load the video capture object on view 1
if cap==None:
    exit()
nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
vidWriter = cv2.VideoWriter("demo1.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480)) #VideoWriter for output
assert(pose2Dv1.shape[0]==pose3Dv0.shape[0])
assert(nF == pose2Dv1.shape[0])
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
        X = int(pose2Dv1[i,3*j+0]) 
        Y = int(pose2Dv1[i,3*j+1])

        pos3Dv0 = pose3Dv0[i,4*j+1:4*j+4] #Retrieve the 3D point at view 0
        #a = pos3Dv0.shape
        pos3Dv1 = originShift(pos3Dv0,P01) #Shifting origin from view 0 to view 1 of 3D point 
        #b = pos3Dv1[2]
        pos3Dv1 =pos3Dv1[:2]/pos3Dv1[2] #Normalize 3D coordinate
        pos2Dx = lst_f[1]*pos3Dv1 + lst_c[1] #Project into 2D point on view1, using intrinsic params of cam 1

        cv2.circle(frame,(X,Y),8,(255,255-j*step,j*step),-1) #Draw the points of OpenPose by big color circle
        cv2.circle(frame,tuple(pos2Dx.astype('int')),4,(0,0,255),-1) #Draw the projected point after origin shifting with smaller red circle
        
    vidWriter.write(frame)
print("")
cap.release() #Good pratice
vidWriter.release() #Good pratice
    
