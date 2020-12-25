import numpy as np
import cv2
import os


class DB2D3DManager:
    def __init__(self,baseDir):
        self.baseDir = baseDir
        self.camDir = os.path.join(baseDir,"camera")
        self.camID = self.loadCamOrder()
        self.nbCam = len(self.camID)
        self.nbJ = 15

    '''
    retrieve a list of camera Index
    '''    
    def loadCamOrder(self,txtFile=None):
        if txtFile==None:
            txtFile="camerasOrder.txt"
        path = os.path.join(self.camDir,txtFile)
        camID=np.loadtxt(path,skiprows=1).astype(int)
        return list(camID)
    
    '''
    Load intrinsic value from dataset for all camera
    output: list lst_f =[[alphaX1, alphaY1],[alphaX2, alphaY2],...]
            list lst_c =[[mx1, my1],[mx2, my2],...]
    '''
    def loadIntrinsic(self):
        lst_f=[]
        lst_c=[]
        for cID in self.camID:
            fs = cv2.FileStorage(os.path.join(self.camDir,"out_camera_data_" + str(cID) +".xml"),cv2.FILE_STORAGE_READ)
            d = fs.getNode("camera_matrix").mat()
            lst_f.append(np.array([d[0,0],d[1,1]]))
            lst_c.append(np.array([d[0,2],d[1,2]]))
        return lst_f, lst_c
    
    '''
    Load the extrinsic matrix (origin shifting matrix)
    input : IdxFrom - index of the first view
            IdxTo ALWAYS NONE
    output : shifting matrix from first view to the (first view + 1) % 4
    '''
    def loadExtrinsic(self,IdxFrom,IdxTo=None):
        IdxTo = (IdxFrom +1) % self.nbCam
        fs = cv2.FileStorage(os.path.join(self.camDir,"pose_" + str(IdxFrom)+ "_" + str(IdxTo) + ".xml"),cv2.FILE_STORAGE_READ)
        d = fs.getNode("transform").mat()
        return d
    
    '''
    Load the 2D pose position detected by OpenPose
    input : subject - subject name
            action - action name
            camIdx - index of camera 
    output: matrix NF x (25*3), NF is number of frame for that sequence
            each line has 75 columns: X1 Y1 F1 ..... X25 Y25 F25 : the coordinates X,Y and reliability score from OpenPose
    '''
    def load2DTXT(self,subject,action,camIdx,OP=True):
        path = os.path.join(self.baseDir,"OP2DTXT",subject,action+"."+str(camIdx)+".txt")
        return np.loadtxt(path)

    '''
    Load the 3D pose after triangulation reference at view 0
    input : subject - subject name
            action - action name
    output: matrix NF x (15*4), NF is number of frame for that sequence
            each line has 60 columns: F1 X1 Y1 Z1 ..... F25 X25 Y25 Z25 : and reliability score and the coordinates X,Y,Z 
    '''
    def load3DTXT(self,subject,action,OP=True):
        path = os.path.join(self.baseDir,"OP3DTXT",subject,action+".txt")
        data = np.loadtxt(path,skiprows =self.nbCam+1)
        data = data[:,self.nbCam*3:]
        nF = int(data.shape[0]/15)
        output = np.zeros((nF,self.nbJ*4))
        for i in range(nF):
            for j in range(self.nbJ):
                output[i,4*j:4*j+4] = data[i*self.nbJ+j]
        return output

    '''
    Load the VideoCapture of one specific sequence
    input : subject - subject name
            action - action name
            camIdx - index of camera 
    output: opencv VideoCapture object
    '''
    def loadVideoCapture(self,subject,action,camIdx):
        path = os.path.join(self.baseDir,"Videos",subject,action+"."+str(camIdx)+".avi")
        cap = cv2.VideoCapture(path)
        if (cap.isOpened()):
            return cap 
        else:
            printf("Can not open video ",path)
            return None
    

