import num as np

def quatInv(quat):
    

    if quat.shape != (4, 1):
        raise ValueError("Input quaternion must be 4x1")

    if abs(np.linalg.norm(quat) - 1) > np.finfo(float).eps:
        raise ValueError("Input quaternion must be unit-length")

    qInv = quat.copy()
    qInv[3,0] = quat[3,0]
    qInv[0:3] = -quat[0:3]

    return qInv