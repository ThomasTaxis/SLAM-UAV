import numpy as np


def convertOxtsToPose(oxts):
    
    scale = np.cos(oxts[0][0] * np.pi / 180.0)  

    pose     = []
    Tr_0_inv = []

    for i in range(len(oxts)):
        
        er = 6378137
        mx = scale * oxts[i][1] * np.pi * er / 180;
        my = scale * er * np.log( np.tan((90+oxts[i][0]) * np.pi / 360))
        
        t = np.zeros((3,1))
        t[0,0] = mx
        t[1,0] = my
        t[2,0] = oxts[i][2]       

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
        
        if (i==0):
          upper_part = np.hstack((R, t))
          lower_part = np.array([[0, 0, 0, 1]])
          matrix = np.vstack((upper_part, lower_part))

          Tr_0_inv = np.linalg.inv(matrix)  
          
          
        upper_part = np.hstack((R, t))
        lower_part = np.array([[0, 0, 0, 1]])
        matrix = np.vstack((upper_part, lower_part))     
            
        pose.append(Tr_0_inv@matrix)
        
    return pose
      
        
        
      
      

