import numpy as np
import os
from load_ts import load_timestamps

def load_oxts_lite_data(base_dir, frames=None):
    """
    Reads GPS/IMU data from files to memory. Requires the base directory
    (=sequence directory) as a parameter. If frames is not specified, loads all frames.
    """
    

    if frames is None:
        ts = load_timestamps(os.path.join(base_dir, 'oxts'))
        oxts = []
        for i, _ in enumerate(ts, start=1):
            file_path = os.path.join(base_dir, f'oxts/data/{i-1:010d}.txt')
            if os.path.exists(file_path):
                oxts.append(np.loadtxt(file_path))
            else:
                oxts.append([])
    else:
        oxts = []
        for i, frame in enumerate(frames, start=1):
            file_name = os.path.join(base_dir, f'oxts/data/{frame-1:010d}.txt')
            try:
                oxts.append(np.loadtxt(file_name))
            except Exception:
                oxts.append([])

    return oxts