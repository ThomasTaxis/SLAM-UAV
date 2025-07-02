import numpy as np

def crossMatToVec(crossMat):
   
    vec = np.array([crossMat[2, 1], crossMat[0, 2], crossMat[1, 0]])
    return vec