from calibration_rigid import load_calibration_rigid
import os
from calibration_cam_to_cam import load_calibration_cam_to_cam
import numpy as np

def load_calibration(calib_dir):
    
    # Load the velodyne to camera calibration
    tr_velo_to_cam = load_calibration_rigid(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    # Load the camera intrinsic and extrinsic calibration
    calib = load_calibration_cam_to_cam(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))

    # Create 4x4 matrix from rectifying rotation matrix
    R_rect00 = np.zeros((4,4))
    R_rect00[:3, :3] = np.array([float(x) for x in calib[8].strip().split(' ')[1:]]).reshape((3,3))
    R_rect00[3,3] = 1
    
    P_rect00 = np.array([float(x) for x in calib[9].strip().split(' ')[1:]]).reshape((3,4))
    
    
    # Compute extrinsics from the first to i'th rectified camera
    T = np.eye(4)
    T[0,3] = P_rect00[0,3]/P_rect00[0,0]
    
    velo_to_cam = T@R_rect00@tr_velo_to_cam

    # Calibration matrix after rectification (equal for all cameras)
    K = P_rect00[:,0:3]

    return velo_to_cam,K