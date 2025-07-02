import numpy as np

def crossMat(vec):
    
    
    if vec.shape != (3, 1):
        raise ValueError("Input vector must be 3x1")

    vecCross = np.array([[0, -vec[2, 0], vec[1, 0]],
                         [vec[2, 0], 0, -vec[0, 0]],
                         [-vec[1, 0], vec[0, 0], 0]])

    return vecCross