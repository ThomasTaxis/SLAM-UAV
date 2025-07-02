import numpy as np
from q_RC import quatRightComp
from q_LC import quatLeftComp

def quatToRotMat(quat):
    
    if quat.shape != (4, 1):
        raise ValueError("Input quaternion must be 4x1")

    # Normalize the quaternion if it's not already unit length
    if not np.isclose(np.linalg.norm(quat), 1, atol=np.finfo(float).eps):
        if np.linalg.norm(quat) - 1 > 0.1:
            print(f"Warning: Input quaternion is not unit-length. norm(q) = {np.linalg.norm(quat)}. Re-normalizing.")
        quat = quat / np.linalg.norm(quat)

    # Compute the rotation matrix
    R = quatRightComp(quat).T @ quatLeftComp(quat)
    C = R[0:3, 0:3]  # Assuming renormalizeRotMat is a normalization step, which might be optional

    return C
