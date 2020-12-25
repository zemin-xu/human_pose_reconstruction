import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits import mplot3d
import math
import cv2
import os
import time
import sys
import argparse
#matplotlib.use('TkAgg')
plt.ioff()


baseDir ="C:/Users/Zemin XU/PycharmProjects/3d_reconstruction/data/"
camDir = os.path.join(baseDir,"camera")
Pose3DPath = baseDir + "OP3DTXTRefined/"
renderPath = baseDir +"OP3DRendered/"

lstParents=[1,8,1,2,3,1,5,6,-1,8,9,10,8,12,13]

def dataloader(file_path):
    metadata = np.loadtxt(file_path,max_rows=1)
    exCamsPara=np.loadtxt(file_path,skiprows=1,max_rows=int(metadata[2]))
    pos3D = np.loadtxt(file_path,skiprows=1+int(metadata[2]))  
    pos3D = pos3D.reshape(-1,int(metadata[0]),3).transpose(0,2,1) 
    return int(metadata[0]),int(metadata[1]),int(metadata[2]),exCamsPara,pos3D


def draw_bone(ax, lstParents, points3D, i=None):
    points3D_ = points3D.copy()
    points3D_ -= points3D_[:,8].reshape(3,1)
    idx_origin=np.array([0,1,2])
    idx_swapped = np.array([0,2,1])
    #Swap y and z for display purpose
    points3D_[idx_origin,:]=points3D_[idx_swapped,:]
    #print("points3D ",points3D_.shape)
    
    
    #Draw right legs
    xrl,yrl,zrl = make_list(lstParents,points3D_,11,8)
    #Draw right arm
    xra,yra,zra = make_list(lstParents,points3D_,4,1)
    #Draw left legs
    xll,yll,zll = make_list(lstParents,points3D_,14,8)
    #Draw left arm
    xla,yla,zla = make_list(lstParents,points3D_,7,1)
    #Draw body
    x,y,z = make_list(lstParents,points3D_,0,8)

    if (i == None):
        ax.clear()
        lstLine.append(ax.plot3D(xrl,yrl,zrl,'red'))
        lstLine.append(ax.plot3D(xra,yra,zra,'red'))
        lstLine.append(ax.plot3D(xll,yll,zll,'blue'))
        lstLine.append(ax.plot3D(xla,yla,zla,'blue'))
        lstLine.append(ax.plot3D(x,y,z,'black'))
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([1, -1])
    else:
        #line.set_xdata(x)
        #line.set_ydata(y)
        #line.set_3d_properties(z)
        #print("Update3D")

        lstLine[5*i-4*nbCam+0][0].set_data_3d(xrl,yrl,zrl)
        lstLine[5*i-4*nbCam+1][0].set_data_3d(xra,yra,zra)
        lstLine[5*i-4*nbCam+2][0].set_data_3d(xll,yll,zll)
        lstLine[5*i-4*nbCam+3][0].set_data_3d(xla,yla,zla)
        lstLine[5*i-4*nbCam+4][0].set_data_3d(x,y,z)
        #ax.draw_artist(ax.patch)
        #ax.draw_artist(line)


def make_list(lstParents, points3D, idx_start,idx_end):
    x=[]
    y=[]
    z=[]
    next_idx = idx_start

    while next_idx != idx_end:
        x.append(points3D[0][next_idx])
        y.append(points3D[1][next_idx])
        z.append(points3D [2][next_idx])
        next_idx = lstParents[next_idx]
   
    x.append(points3D[0][next_idx])
    y.append(points3D[1][next_idx])
    z.append(points3D [2][next_idx])
    return x,y,z

def animate(nframe):
    global cp0,lstLine
    #fig = plt.figure(figsize=plt.figaspect(1/5))  
    #plt.clf()
    sys.stdout.write('\r')
    sys.stdout.write("{}/{}".format(nframe,nF))
    step = 255.0/(nJ-1) 
    #Draw keypoints on images
    #print("Start new cycles: ", time.time()-cp0)
    if (nframe==0):
        lstLine=[]
    for i in range(len(lstCap)):
        lstCap[i].set(cv2.CAP_PROP_POS_FRAMES,nframe)
        ret, frame = lstCap[i].read()
       
        XX = projectedPointsCameras[i]
        for j in range(nJ):
            cv2.circle(frame,tuple(XX[nframe,:,j].astype('int')),8,(255,255,j*step),-1)
        frame = frame[:,:,::-1]
      
        
        if(nframe == 0):
            lstAxes[i].clear()
            lstLine.append(lstAxes[i].imshow(frame))
            lstAxes[i].set_title("View "+str(i+1))
            lstAxes[i].set_axis_off()
        else:
            lstLine[i].set_data(frame)
        #plt.draw

    #print("Render images: ", time.time()-cp0)
    #Draw 3D pose
    #ax = fig.add_subplot(1,5,5,projection='3d')
    if (nframe==0):
        draw_bone(lstAxes[baseAxe],lstParents,pose3Dorigin[nframe,:,:])
        draw_bone(lstAxes[baseAxe+1],lstParents,pose3D[nframe,:,:])
        lstAxes[baseAxe].set_title("Raw SfM")
        lstAxes[baseAxe+1].set_title("Optimizied")
    else:
        draw_bone(lstAxes[baseAxe],lstParents,pose3Dorigin[nframe,:,:],baseAxe)
        draw_bone(lstAxes[baseAxe+1],lstParents,pose3D[nframe,:,:],baseAxe+1)
    
    #plt.draw()
    #plt.pause(0.01)
    #print("Render bones: ",time.time()-cp0)
    cp0=time.time()

def parse_args():
    parser = argparse.ArgumentParser(description='Render script')
    parser.add_argument('-s', '--subject', default='Lea', type=str, metavar='NAME', help='specific subject to be rendered') 
    parser.add_argument('-a', '--action', default='', type=str, metavar='NAME', help='specific action to be rendered') 
    return  parser.parse_args()

#Start here

args = parse_args()
chosenSubject = args.subject
chosenAction = args.action
foundSubject=False
foundAction = False
lstSubject=[]
with open(os.path.join(baseDir,"metaperson.txt"),"r") as f:
    lstSubject = f.read().splitlines()
camID=np.loadtxt(os.path.join(camDir,"camerasOrder.txt"),skiprows=1).astype(int)
camID=list(camID)
nbCam = len(camID)
baseAxe = nbCam 
nbAxe = nbCam +2

#Load intrinsic camera params
lst_f=[]
lst_c=[]
for cID in camID:
    fs = cv2.FileStorage(os.path.join(camDir,"out_camera_data_" + str(cID) +".xml"),cv2.FILE_STORAGE_READ)
    d = fs.getNode("camera_matrix").mat()
    lst_f.append(np.array([d[0,0],d[1,1]]))
    lst_c.append(np.array([d[0,2],d[1,2]]))

for subject in lstSubject:
    if (subject != chosenSubject and chosenSubject !=''):
        continue
    foundSubject=True
    #Subject_Path=os.path.join(baseDir,"Videos",subject)
    Subject_Path = baseDir + "Videos/" + subject
    lstActions= []
    with open(os.path.join(Subject_Path,"metaactivity.txt"),"r") as f:
        lstActions = f.read().splitlines()
    
    for action in lstActions:
        if (action != chosenAction and chosenAction !=''):
            continue
        foundAction = True
        print("Rendering {} | {}".format(subject,action))
        lstCap=[]
        for cam in camID:
            lstCap.append(cv2.VideoCapture(os.path.join(Subject_Path,action+"."+str(cam)+".avi")))           
        #Load 3D da
        print("Reading 3D pose")
        nJ,nF,nC,exCamsPara,pose3D=dataloader(os.path.join(Pose3DPath,subject,action+ "_refined.txt"))
        _,_,_,_,pose3Dorigin=dataloader(os.path.join(Pose3DPath,subject,action+ "_origin.txt"))
        fig = plt.figure(figsize=(12+6*nbCam, 6)) 
        lstAxes=[]
        lstLine=[]
        projectedPointsCameras=[]
        print("Projecting points into image")
        for i in range(len(lstCap)):
            r= exCamsPara[i][:3]
            T= np.expand_dims(exCamsPara[i][3:],axis=0).transpose(1,0)
            f=np.expand_dims(lst_f[i],axis=0).transpose(1,0)
            c=np.expand_dims(lst_c[i],axis=0).transpose(1,0)
            R,_=cv2.Rodrigues(r)
        
            X = np.dot(R,pose3D).transpose(1,0,2)+T # F x

            X = X[:, :2, :]/np.tile(X[:, 2:3, :]+1e-8, (1, 2, 1))
            XX = X*f+c  # Project to image plane
            projectedPointsCameras.append(XX.copy())
            lstAxes.append(fig.add_subplot(1, nbAxe, i+1))

        lstAxes.append(fig.add_subplot(1,nbAxe,baseAxe+1,projection='3d'))

        lstAxes.append(fig.add_subplot(1,nbAxe,baseAxe+2,projection='3d'))
        cp0 = time.time()
        print("Start saving gif...")     
        
        anim = FuncAnimation(fig, animate, frames=np.arange(0, nF), interval=1000/30, repeat=False)
        #anim.save(os.path.join(renderPath,subject,action+".mp4"), dpi=80, writer='ffmpeg')
        res_path = renderPath + subject + action + ".mp4"
        anim.save(res_path, dpi=80, writer='ffmpeg')
        plt.close(fig)
        sys.stdout.write("\n")
        print("Saved file ", res_path)

