import numpy as np

def latToScale(lat):
    scale = np.cos(lat * np.pi / 180.0);
    
    return scale