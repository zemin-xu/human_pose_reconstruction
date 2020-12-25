import numpy as np 
import math
import cv2
from scipy.linalg import null_space

def decomposeRQ(M):
    R=M;
    denom = math.sqrt(R[2,1]*R[2,1]+R[2,2]*R[2,2])
    s = -R[2,1]/denom
    c = R[2,2]/denom
    Qx = np.array(((1, 0, 0),(0, c, -s),( 0, s, c)))
    R=np.dot(R,Qx)

    denom = math.sqrt(R[2,0]*R[2,0]+R[2,2]*R[2,2]);
    s = R[2,0]/denom
    c = R[2,2]/denom
    Qy = np.array(((c, 0, s),( 0, 1, 0), (-s, 0, c)));
    R=np.dot(R,Qy);
    
    denom = math.sqrt(R[1,0]*R[1,0]+R[1,1]*R[1,1])
    s = -R[1,0]/denom;
    c = R[1,1]/denom;
    Qz = np.array(((c, -s, 0),(s, c, 0),( 0, 0, 1)));
    R=np.dot(R,Qz);
    
    R=R/R[2,2];
    Q= np.dot(Qx,np.dot(Qy,Qz)).transpose();
    return R,Q

P =np.array(((-0.0343,-0.0043, -0.99, 3.89),(-0.158, 0.987, 0.0012 , 0.231),(0.986, 0.158, -0.0345,3.582)))
P=P*2
M = P[:,:3]
B = P[:,3]
xn = null_space(P)
x = np.linalg.solve(M, -B)
R,Q = decomposeRQ(M)
print(R)
print(Q)
RR=np.zeros((3,3))
QQ=np.zeros((3,3))
a= cv2.RQDecomp3x3(M,RR,QQ)
print(RR)
print(QQ)
print(x)
print(xn/xn[3,0])
