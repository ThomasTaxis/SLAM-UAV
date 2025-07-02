import numpy as np

from crossVec import crossMat

def axisAngleToRotMat(psi):
    """
    Converts an axis-angle rotation vector into a rotation matrix.

    :param psi: axis-angle rotation vector
    :return: Rotation matrix
    """
    np_norm = np.linalg.norm(psi)
    cp = np.cos(np_norm)
    sp = np.sin(np_norm)
    pnp = psi / np_norm
    pnp = pnp.reshape((3,1))
    
    Psi = cp * np.eye(3) + (1 - cp) * (pnp@pnp.T) - sp * crossMat(pnp)
    return Psi
