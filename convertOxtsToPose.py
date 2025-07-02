import numpy as np
from latToScale import latToScale
from latlonToMercator import latlonToMercator

def convertOxtsToPose(oxts):
    scale = latToScale(oxts[0][0])
    
    pose     = []
    
    Tr_0_inv = None


    for i in range(len(oxts)):
        
        if not oxts:
            pose[i] = None  # Or an empty list, depending on your needs
            continue 
        
        t = np.zeros((3,1))
        
        t[0,0] ,t[1,0] = latlonToMercator(oxts[i][0],oxts[i][1],scale);
        t[2,0] = oxts[i][2];
        
        rx = oxts[i][3]
        ry = oxts[i][4]
        rz = oxts[i][5]
        Rx = np.array([[1, 0, 0],
                  [0, np.cos(rx), -np.sin(rx)],
                  [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                  [np.sin(rz), np.cos(rz), 0],
                  [0, 0, 1]])
        R  = Rz@Ry@Rx
        
        if Tr_0_inv is None or not Tr_0_inv.size:
            transformation_matrix = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
            Tr_0_inv = np.linalg.inv(transformation_matrix)
            
        pose.append(Tr_0_inv @ np.vstack((np.hstack((R, t)), [0, 0, 0, 1])))
       
            
        
    
    
    return pose