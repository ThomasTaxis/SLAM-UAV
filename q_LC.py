from crossVec import crossMat
import numpy as np

def quatLeftComp(quat):
    
    if quat.shape != (4, 1):
        raise ValueError("Input quaternion must be 4x1")
        
    vector = quat[0:3]
    scalar = quat[3,0]

    # Use the crossMat function to create the cross-product matrix for the vector part
    cross_mat = crossMat(vector)

    # Constructing the left compound matrix
    qLC = np.zeros((4, 4))
    qLC[0:3, 0:3] = scalar * np.eye(3) - cross_mat
    qLC[0:3, 3] = vector[:, 0]
    qLC[3, 0:3] = -vector.T
    qLC[3, 3] = scalar

    return qLC