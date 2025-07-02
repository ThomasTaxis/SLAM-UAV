from crossVec import crossMat
import numpy as np

def quatRightComp(quat):
    

    if quat.shape != (4, 1):
        raise ValueError("Input quaternion must be 4x1")

    vector = quat[0:3]
    scalar = quat[3, 0]  # Ensuring scalar is a true scalar

    # Use the crossMat function to create the cross-product matrix for the vector part
    cross_mat = crossMat(vector)

    # Constructing the right compound matrix
    qRC = np.zeros((4, 4))
    qRC[0:3, 0:3] = scalar * np.eye(3) + cross_mat
    qRC[0:3, 3] = vector[:, 0]
    qRC[3, 0:3] = -vector.T
    qRC[3, 3] = scalar

    return qRC