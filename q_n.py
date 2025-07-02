import numpy as np

def quat_normalize(q):
    
    
    if q.shape != (4, 1):
        raise ValueError("Input quaternion must be 4x1")

    n = np.sqrt(q.T@q)
    qn = q / n
    return qn
