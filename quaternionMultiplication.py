import numpy as np
from q_LC import quatLeftComp

def quat_mult(quat1, quat2):
    """
    Multiplies two quaternions using the {i,j,k,1} convention.
    
    Parameters:
    quat1 (np.array): First quaternion as a 4x1 numpy array.
    quat2 (np.array): Second quaternion as a 4x1 numpy array.
    
    Returns:
    np.array: The product of the two quaternions.
    """
    if quat1.shape != (4, 1) or quat2.shape != (4, 1):
        raise ValueError("Input quaternions must be 4x1")

    # Define the left composition matrix for quaternion multiplication
    q_prod = quatLeftComp(quat1)@quat2
    
    return q_prod
