from IMUnormalize import convertOxtsToPose
import numpy as np
from  load_oxts import load_oxts_lite_data


def getGroundTruth(filename,IMUframes):
    
    oxts = load_oxts_lite_data(filename,frames=None)
    
    pose = convertOxtsToPose(oxts)
    
    poseMat = np.zeros((4,4,len(IMUframes)))
    
    for i in range(len(IMUframes)):
        if (i == 0):
            firstPose = pose[IMUframes[i]]
            
        poseMat[:,:,i] = np.linalg.inv(firstPose)@pose[IMUframes[i]]
        
        
    return poseMat