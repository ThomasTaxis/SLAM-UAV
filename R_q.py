import numpy as np

def rotMatToQuat(R):
    

    r, c = R.shape
    R = R.T

    if r != 3 or c != 3:
        print('R must be a 3x3 matrix')
        return None

    Rxx, Rxy, Rxz = R[0, 0], R[0, 1], R[0, 2]
    Ryx, Ryy, Ryz = R[1, 0], R[1, 1], R[1, 2]
    Rzx, Rzy, Rzz = R[2, 0], R[2, 1], R[2, 2]

    w = np.sqrt(np.trace(R) + 1) / 2

    # Check if w is real. If not, set it to zero.
    if np.iscomplex(w):
        w = 0

    x = np.sqrt(1 + Rxx - Ryy - Rzz) / 2
    y = np.sqrt(1 + Ryy - Rxx - Rzz) / 2
    z = np.sqrt(1 + Rzz - Ryy - Rxx) / 2

    max_idx = np.argmax([w, x, y, z])

    if max_idx == 0:
        x = (Rzy - Ryz) / (4 * w)
        y = (Rxz - Rzx) / (4 * w)
        z = (Ryx - Rxy) / (4 * w)
    elif max_idx == 1:
        w = (Rzy - Ryz) / (4 * x)
        y = (Rxy + Ryx) / (4 * x)
        z = (Rzx + Rxz) / (4 * x)
    elif max_idx == 2:
        w = (Rxz - Rzx) / (4 * y)
        x = (Rxy + Ryx) / (4 * y)
        z = (Ryz + Rzy) / (4 * y)
    elif max_idx == 3:
        w = (Ryx - Rxy) / (4 * z)
        x = (Rzx + Rxz) / (4 * z)
        y = (Ryz + Rzy) / (4 * z)

    return np.array([[x], [y], [z], [w]])