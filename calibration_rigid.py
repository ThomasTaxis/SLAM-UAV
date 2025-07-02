import numpy as np

def load_calibration_rigid(filename):
    """
    Loads a rigid calibration from a file, forming a transformation matrix.
    """
    with open(filename, 'r') as f:
        Rt = f.readlines()
        
    R = np.array([float(x) for x in Rt[1].strip().split(' ')[1:]]).reshape((3,3))
    T = np.array([float(x) for x in Rt[2].strip().split(' ')[1:]]).reshape((3,1))
    
    Tr = np.eye(4)
    Tr[:3, :3] = R
    Tr[:3, 3] = T.flatten()

        # Construct the transformation matrix
          # Convert T from 3x1 to 1D array for assignment

    return Tr
