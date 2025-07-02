import numpy as np

def rotmat_from_quat(q):
    

    if q.shape != (4, 1):
        raise ValueError("Input quaternion must be 4x1")

    b2 = q[1, 0] ** 2
    c2 = q[2, 0] ** 2
    d2 = q[3, 0] ** 2

    R = np.array([
        [1 - 2 * c2 - 2 * d2, 2 * q[1, 0] * q[2, 0] - 2 * q[0, 0] * q[3, 0], 2 * q[0, 0] * q[2, 0] + 2 * q[1, 0] * q[3, 0]],
        [2 * q[1, 0] * q[2, 0] + 2 * q[0, 0] * q[3, 0], 1 - 2 * b2 - 2 * d2, 2 * q[2, 0] * q[3, 0] - 2 * q[0, 0] * q[1, 0]],
        [2 * q[1, 0] * q[3, 0] - 2 * q[0, 0] * q[2, 0], 2 * q[0, 0] * q[1, 0] + 2 * q[2, 0] * q[3, 0], 1 - 2 * b2 - 2 * c2]
    ])

    return R