import cv2
import numpy as np
import os
import glob
import numpy as np
import os
from scipy.linalg import null_space
from  load_oxts import load_oxts_lite_data
from IMUnormalize import convertOxtsToPose
from load_ts import load_timestamps
from calibration_cam_to_cam import load_calibration_cam_to_cam
from load_calibration1 import load_calibration
from calibration_rigid import load_calibration_rigid
from R_q import rotMatToQuat
from getGroundTruth import getGroundTruth
from Hn import Jacobian
from axisRotMat import axisAngleToRotMat
from qToR import quatToRotMat
from q_RC import quatRightComp
from q_LC import quatLeftComp
from FeatPos import triangulate, calc_gn_pos_est
from ReprojError import calc_residual
from crossVec import crossMat
from scipy.linalg import null_space
from QRdecomposition import T_H
from MatToVec import crossMatToVec
import matplotlib.pyplot as plt
from quaternionMultiplication import quat_mult

image_data = r'C:\Users\ttaxi\Thesis\2011_09_29_drive_0071_sync\2011_09_29\2011_09_29_drive_0071_sync'


image_files = [f for f in os.listdir(image_data) if f.endswith(('.png', '.jpg', '.jpeg'))]


image_files = image_files[::5]

    

    
images = []

sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck= True)

images_RGB = []

for filename in image_files:
    image_path = os.path.join(image_data, filename)

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(gray)        
    images_RGB.append(image)
    
images = np.array(images)   




keypoints_info = {}

global_keypoint_index = 0

descriptors = []  # Store descriptors for each image for later matching

for idx, filename in enumerate(image_files):  
    gray = images[idx] 

    keypoints, des  = sift.detectAndCompute(gray, None)
    
    keypoints_info[filename] = [((kp.pt[0]), (kp.pt[1]), global_keypoint_index + i) for i, kp in enumerate(keypoints)]
    
    global_keypoint_index += len(keypoints)
    
    descriptors.append((filename, des))
    
matches_info = {}  

for i in range(1, len(descriptors)):
    filename1, des1 = descriptors[i-1]
    filename2, des2 = descriptors[i]
    
    matches = bf.match(des1, des2)

    # Filter matches using the Lowe's ratio test
    #ratio_thresh = 0.5
    #good_matches = []
    #for m, n in matches:
     #   if m.distance < ratio_thresh * n.distance:
      #      good_matches.append(m)
            
            
    matches = sorted(matches, key=lambda x: x.distance)
    



    
    matches_info[(filename1, filename2)] = []
    
    
    for match in matches:
        
        curr_keypoint_info = keypoints_info[filename1][match.queryIdx]
        next_keypoint_info = keypoints_info[filename2][match.trainIdx]
        curr_id = curr_keypoint_info[2]
        next_id = next_keypoint_info[2]

        curr_coord = (curr_keypoint_info[0], curr_keypoint_info[1])
        next_coord = (next_keypoint_info[0], next_keypoint_info[1])
        
        

        matches_info[(filename1, filename2)].append((curr_id, curr_coord, next_id, next_coord))


#%%
inliers_info = {}  
outliers_info = {}

rejected_set = set() 

for image_pair, matches in matches_info.items():
    src_pts = np.float32([match[1] for match in matches])
    dst_pts = np.float32([match[3] for match in matches])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)

    # Update inliers and outliers, and populate rejected_set
    inliers = []
    outliers = []
    for match, m in zip(matches, mask.ravel()):
        if m == 1:
            inliers.append(match)
        else:
            outliers.append(match)
            rejected_set.add(match[0])  
            rejected_set.add(match[2])  
    
    inliers_info[image_pair] = inliers
    outliers_info[image_pair] = outliers
    
filtered_inliers_info = {}

for image_pair, matches in inliers_info.items():
    filtered_matches = [match for match in matches if match[0] not in rejected_set and match[2] not in rejected_set]
    
    filtered_inliers_info[image_pair] = filtered_matches
    
    
common_ids_check = set()  

for image_pair, matches in filtered_inliers_info.items():
    for match in matches:
        if match[0] in rejected_set or match[2] in rejected_set:
            common_ids_check.add(match[0])
            common_ids_check.add(match[2])

if common_ids_check:
    print(f"IDs in both filtered inliers and rejected_set: {common_ids_check}")
else:
    print("No common IDs between filtered inliers and rejected_set.")
    


filtered_dict = {}

def find_group_and_key(keypoint_id, image_filename):
    for group_key, group in filtered_dict.items():
        if any(kp_id == keypoint_id and img_filename == image_filename for img_filename, kp_id, _ in group):
            return group, group_key
    return None, None

# Iterate through the matches_info dictionary
for (first_image_filename, second_image_filename), matched_info in filtered_inliers_info.items():
    for curr_id, curr_coord, next_id, next_coord in matched_info:
        # Find the group and its key for the current and next keypoints
        curr_group, curr_group_key = find_group_and_key(curr_id, first_image_filename)
        next_group, next_group_key = find_group_and_key(next_id, second_image_filename)

        if curr_group is not None and next_group is not None:
            if curr_group != next_group:
                # Merge the groups if both keypoints are already in different groups
                merged_group = curr_group + next_group
                filtered_dict[curr_group_key] = merged_group
                
          
                
        elif curr_group is not None:
            # Add the next keypoint to the current keypoint's group if it's not already present
            if (second_image_filename, next_id, next_coord) not in curr_group:
                curr_group.append((second_image_filename, next_id, next_coord))
                filtered_dict[curr_group_key] = curr_group
        elif next_group is not None:
            # Add the current keypoint to the next keypoint's group if it's not already present
            if (first_image_filename, curr_id, curr_coord) not in next_group:
                next_group.append((first_image_filename, curr_id, curr_coord))
                filtered_dict[next_group_key] = next_group
        else:
            # Create a new group with both keypoints
            new_group_key = len(filtered_dict)
            filtered_dict[new_group_key] = [
                (first_image_filename, curr_id, curr_coord),
                (second_image_filename, next_id, next_coord)
                ]
 
image_files = [f for f in os.listdir(image_data) if f.endswith(('.png', '.jpg', '.jpeg'))]


oxts = load_oxts_lite_data(r'C:\Users\ttaxi\Thesis\2011_09_29_drive_0071_sync\2011_09_29\2011_09_29_drive_0071_sync')

Rt_IMUtoG = convertOxtsToPose(oxts)

v = []

for i in range(len(oxts)):
    v.append(oxts[i][8:11])
    

a = []

for i in range(len(oxts)):
    a.append(oxts[i][14:17])


w = []

for i in range(len(oxts)):
    w.append(oxts[i][20:23])
    

    

image_timestamps = load_timestamps(r'C:\Users\ttaxi\Thesis\2011_09_29_drive_0071_sync\2011_09_29\2011_09_29_drive_0071_sync',True)
IMU_timestamps = load_timestamps(r'C:\Users\ttaxi\Thesis\2011_09_29_drive_0071_sync\2011_09_29\2011_09_29_drive_0071_sync\oxts',True)




T_VeloToCam,K = load_calibration(r'C:\Users\ttaxi\Thesis\2011_09_29_calib\2011_09_29')

T_IMUtoVelo = load_calibration_rigid(r'C:\Users\ttaxi\Thesis\2011_09_29_calib\2011_09_29\calib_imu_to_velo.txt')

T_IMUtoCam = T_VeloToCam@T_IMUtoVelo

T_CamtoIMU = np.linalg.inv(T_IMUtoCam)


    
q_CI = rotMatToQuat(T_IMUtoCam[:3,:3])

p_C_I = T_CamtoIMU[:3,3]

C_CI = T_IMUtoCam[:3,:3]
q_IC = rotMatToQuat(C_CI.T)
p_I_C = T_IMUtoCam[:3,3]                                                                                                                                              

    


Rt_CamtoG = []
for i in range(len(Rt_IMUtoG)):
    
    Rt_CamtoG.append(Rt_IMUtoG[i]@np.linalg.inv(T_IMUtoCam))
    


cu = K[0,2];
cv = K[1,2];
fu = K[0,0];
fv = K[1,1];
r_i_vk_i = np.empty((3, len(Rt_IMUtoG)))
r_i_vk_i[:] = np.NaN
theta_vk_i = np.empty((3, len(Rt_IMUtoG)))
theta_vk_i[:] = np.NaN

for j in range(len(Rt_IMUtoG)):
    r_i_vk_i[:, j] = Rt_IMUtoG[j][0:3, 3]
    R_wIMU = Rt_IMUtoG[j][0:3, 0:3]
    theta = np.arccos((np.trace(R_wIMU) - 1) * 0.5)
    if theta < np.finfo(float).eps:
        w1 = np.zeros(3)
    else:
        w1 = 1 / (2 * np.sin(theta)) * np.array([R_wIMU[2, 1] - R_wIMU[1, 2],
                                                 R_wIMU[0, 2] - R_wIMU[2, 0],
                                                 R_wIMU[1, 0] - R_wIMU[0, 1]])
    theta_vk_i[:, j] = theta * w1

groundTruthStates = [{} for _ in range(len(a))]

for state_k in range(len(a)):
    
    if (state_k==0):
        
        q_IG = rotMatToQuat(np.eye(3)) 
        p_I_G = r_i_vk_i[:, state_k]
    else:
        q_IG = rotMatToQuat(axisAngleToRotMat(theta_vk_i[:, state_k]))
        p_I_G = r_i_vk_i[:, state_k]     

        
    

    groundTruthStates[state_k]['imuState'] = {'q_IG': q_IG, 'p_I_G': p_I_G}

    C_IG = quatToRotMat(q_IG)
    q_CG = quatLeftComp(q_CI) @ q_IG  
    p_C_G = p_I_G + C_IG.T @ p_C_I

    groundTruthStates[state_k]['camState'] = {'q_CG': q_CG, 'p_C_G': p_C_G}

y_var = 11**2* np.ones((1,4))              
u_var_prime = y_var[0,0]/fu**2
v_var_prime = y_var[0,1]/fv**2

a_bias = a[669:697]
            
b_a = np.mean(a_bias,axis=0) + np.array([0,0,-9.81])

g_bias = w[669:697]
            
b_g = np.mean(g_bias,axis=0)

omega = np.array(w)

omega_m = omega - b_g

omega_m_std = np.std(omega_m, axis=0)
n_g = np.random.normal(0, omega_m_std, omega_m.shape)

alpha = np.array(a)

alpha_m = alpha - b_a

alpha_m_std = np.std(alpha_m, axis=0)
n_a = np.random.normal(0, alpha_m_std, alpha_m.shape)

image_files = image_files[::5]
v = v[::5]
a = a[::5]
w = w[::5]
Rt_IMUtoG = Rt_IMUtoG[::5]
image_timestamps = image_timestamps[::5]
IMU_timestamps = IMU_timestamps[::5]
diff_t_Cam = np.diff(image_timestamps)
dT = np.append(0, diff_t_Cam)
groundTruthStates = groundTruthStates[::5]
n_g = n_g[::5,:]
n_a = n_a[::5,:]

poses4 = {k: v for k, v in filtered_dict.items() if len(v) > 4 }


EKF_update = []


image_to_index = {f"{i:010}.png": i // 5 for i in range(0, 1059, 5)}

for frame_idx in range(max(image_to_index.values()) + 1):
    current_frame_features = []  # To store features ending at the current index

    # Iterate through your dictionary to find features that end at the current index
    for feature_number, details in poses4.items():
        # Check if the last appearance image of the feature is in the selected images
        if image_to_index.get(details[-1][0]) == frame_idx:
            # Append the feature number along with its details
            current_frame_features.append((feature_number, details))
            
    EKF_update.append(current_frame_features)
    

q_var_init = 0.19306289318771971*np.ones((1,3))  
p_var_init = oxts[0][23]*np.ones((1,3))        
bg_var_init = (omega_m_std**2 ).reshape((1,3))      
ba_var_init = (alpha_m_std**2 ).reshape((1,3))  
v_var_init = oxts[0][24]*np.ones((1,3))       

Q = np.eye(12)
Q[:3, :3] *= omega_m_std*np.ones((1,3))
Q[3:6, 3:6] *= b_g*np.ones((1,3))
Q[6:9, 6:9] *= alpha_m_std*np.ones((1,3))
Q[9:, 9:] *= b_a*np.ones((1,3))

P_IMU_0 = np.diagflat(np.concatenate((q_var_init,bg_var_init,v_var_init,ba_var_init,p_var_init)))



measurements = {}

for state_k in range(len(image_files)):
    measurements[state_k] = {
        'dT': dT[state_k],  # sampling times
        'omega': w[state_k],  # angular velocity
        'v': v[state_k],  # linear velocity
        'a' : a[state_k]
    }
    
for state_k in range(len(image_files)):
    frame_number = image_files[state_k]
    camera_measurements = []

    for feature_id, observations in poses4.items():
        for observation in observations:
            if observation[0] == frame_number:
                camera_measurements.append({'id': feature_id, 'coords': observation[2]})

    measurements[state_k]['camera_measurements'] = camera_measurements
    

#%% Initial state

C_CI = quatToRotMat(q_CI)

bias_a = []
bias_g = []
q1 = []
p1 = []

q_IG = groundTruthStates[0]['imuState']['q_IG']
p_I_G = groundTruthStates[0]['imuState']['p_I_G']






C_IG = quatToRotMat(q_IG)

q_CG = groundTruthStates[0]['camState']['q_CG']
p_C_G = groundTruthStates[0]['camState']['p_C_G']

P_CAM_0 = np.zeros((6,6))

P_CAM_IMU_0 = np.zeros((15,6))

P = P_IMU_0







J = np.zeros((6, 15))
J[0:3,0:3] = C_CI;
J[3:6,0:3] = crossMat((C_IG.T@ p_C_I).reshape((3,1)))
J[3:6,12:15] = np.eye(3)
    
    
tempMat = np.vstack((np.eye(15),J))

P_aug = tempMat @ P @ tempMat.T

P_IMU_aug = P_aug[0:15,0:15]
P_CAM_aug = P_aug[15:21,15:21]
P_CAM_IMU_aug = P_aug[0:15,15:21]

q1.append(q_CG)
p1.append(p_C_G)
bias_a.append(b_a)
bias_g.append(b_g)

#%%

for i in range(1,len(EKF_update)):
    
    if not EKF_update[i]:
        
        q_IG = groundTruthStates[i]['imuState']['q_IG']
        p_I_G = groundTruthStates[i]['imuState']['p_I_G']
        
        C_IG = quatToRotMat(q_IG)

        q_CG = groundTruthStates[i]['camState']['q_CG']
        p_C_G = groundTruthStates[i]['camState']['p_C_G']
        
        F = np.zeros((15, 15))

        omegaHat = measurements[i]['omega'] - b_g.reshape((3,)) - n_g[i,:]
        omegaHat = omegaHat.reshape((3,1))
        aHat = measurements[i]['a'] - b_a.reshape((3,)) - n_a[i,:]
        aHat = aHat.reshape((3,1))
        

        F[0:3, 0:3] = -crossMat(omegaHat)
        F[0:3, 3:6] = -np.eye(3)
        F[6:9, 0:3] = -C_IG.T @ crossMat(aHat)
        F[6:9, 9:12] = -C_IG.T
        F[12:15,6:9] = np.eye(3)

        G = np.zeros((15,12))

        G[0:3,0:3] = -np.eye(3)
        G[3:6,3:6] = np.eye(3)
        G[9:12,9:12] = np.eye(3)
        G[6:9,6:9] = -C_IG.T

        Phi = np.eye((F.shape[0])) + F * dT[i] + F * dT[i] @ F * dT[i]/2. + F * dT[i] @ F * dT[i] @ F * dT[i] /6.
        
        R_kk_1 = quatToRotMat(np.array([[0],[0],[0],[1]]))
        Phi[:3, :3] = quatToRotMat(q_IC) @ R_kk_1.T

        u = R_kk_1 @ (np.array([0,0,-9.81])).reshape((3,1))
       
        s = u/np.linalg.norm(u)**2

        A1 = Phi[6:9, :3]
        w1 = crossMat(np.array([[0],[0],[0]]) - measurements[i]['v'].reshape((3,1))) @ np.array([0,0,-9.81]).reshape((3,1))
        Phi[6:9, :3] = A1 - (A1 @ u - w1) * s

        A2 = Phi[12:15, :3]
        w2 = crossMat(dT[i]*measurements[i]['v'].reshape((3,1))+np.array([[0],[0],[0]]) - 
            p_I_G.reshape((3,1))) @ np.array([0,0,-9.81]).reshape((3,1))
        Phi[12:15, :3] = A2 - (A2 @ u - w2) * s

        Q1 = Phi @ G @ Q @ G.T @ Phi.T * dT[i]
        P_IMU = Phi @ P_IMU_aug @ Phi.T + Q1
        P_IMU = (P_IMU + P_IMU.T)/2
            
      
        P_CAM = P_CAM_aug

        P_CAM_IMU = P_CAM_IMU_aug

       




        P  = np.block([
            [P_IMU, P_CAM_IMU],
            [P_CAM_IMU.T, P_CAM]  
        ])




        J = np.zeros((6, 15+6*i))
        J[0:3,0:3] = C_CI;
        J[3:6,0:3] = crossMat((C_IG.T@ p_C_I).reshape((3,1)))
        J[3:6,12:15] = np.eye(3)
            
            
        tempMat = np.vstack((np.eye(15+6*i),J))

        P_aug = tempMat @ P @ tempMat.T

        P_IMU_aug = P_aug[0:15,0:15]
        P_CAM_aug = P_aug[15:P_aug.shape[0],15:P_aug.shape[1]]
        P_CAM_IMU_aug = P_aug[0:15,15:P_aug.shape[1]]
        
       
        
        p1.append(p_C_G)
        q1.append(q_CG)
        bias_a.append(b_a)
        bias_g.append(b_g)
        
    else:
        q_IG = groundTruthStates[i]['imuState']['q_IG']
        p_I_G = groundTruthStates[i]['imuState']['p_I_G']
        
        C_IG = quatToRotMat(q_IG)

        q_CG = groundTruthStates[i]['camState']['q_CG']
        p_C_G = groundTruthStates[i]['camState']['p_C_G']
        
        F = np.zeros((15, 15))

        omegaHat = measurements[i]['omega'] - b_g.reshape((3,)) - n_g[i,:]
        omegaHat = omegaHat.reshape((3,1))
        aHat = measurements[i]['a'] - b_a.reshape((3,)) - n_a[i,:]
        aHat = aHat.reshape((3,1))
        

        F[0:3, 0:3] = -crossMat(omegaHat)
        F[0:3, 3:6] = -np.eye(3)
        F[6:9, 0:3] = -C_IG.T @ crossMat(aHat)
        F[6:9, 9:12] = -C_IG.T
        F[12:15,6:9] = np.eye(3)

        G = np.zeros((15,12))

        G[0:3,0:3] = -np.eye(3)
        G[3:6,3:6] = np.eye(3)
        G[9:12,9:12] = np.eye(3)
        G[6:9,6:9] = -C_IG.T


        Phi = np.eye((F.shape[0])) + F * dT[i] + F * dT[i] @ F * dT[i]/2. + F * dT[i] @ F * dT[i] @ F * dT[i] /6.
        
        R_kk_1 = quatToRotMat(np.array([[0],[0],[0],[1]]))
        Phi[:3, :3] = quatToRotMat(q_IC) @ R_kk_1.T

        u = R_kk_1 @ (np.array([0,0,-9.81])).reshape((3,1))
        # s = (u.T @ u).inverse() @ u.T
        # s = np.linalg.inv(u[:, None] * u) @ u
        s = u/np.linalg.norm(u)**2

        A1 = Phi[6:9, :3]
        w1 = crossMat(np.array([[0],[0],[0]]) - measurements[i]['v'].reshape((3,1))) @ np.array([0,0,-9.81]).reshape((3,1))
        Phi[6:9, :3] = A1 - (A1 @ u - w1) * s

        A2 = Phi[12:15, :3]
        w2 = crossMat(dT[i]*measurements[i]['v'].reshape((3,1))+np.array([[0],[0],[0]]) - 
            p_I_G.reshape((3,1))) @ np.array([0,0,-9.81]).reshape((3,1))
        Phi[12:15, :3] = A2 - (A2 @ u - w2) * s

        Q1 = Phi @ G @ Q @ G.T @ Phi.T * dT[i]
        P_IMU = Phi @ P_IMU_aug @ Phi.T 
        P_IMU = (P_IMU + P_IMU.T)/2
            
       

        P_CAM = P_CAM_aug

        P_CAM_IMU = P_CAM_IMU_aug
        
      

        P_CAM = P_CAM_aug

        P_CAM_IMU = P_CAM_IMU_aug

        P  = np.block([
            [P_IMU, P_CAM_IMU],
            [P_CAM_IMU.T, P_CAM] 
        ])




        J = np.zeros((6, 15+6*i))
        J[0:3,0:3] = C_CI;
        J[3:6,0:3] = crossMat((C_IG.T@ p_C_I).reshape((3,1)))
        J[3:6,12:15] = np.eye(3)
            
            
        tempMat = np.vstack((np.eye(15+6*i),J))

        P_aug = tempMat @ P @ tempMat.T

        P_IMU_aug = P_aug[0:15,0:15]
        P_CAM_aug = P_aug[15:P_aug.shape[0],15:P_aug.shape[1]]
        P_CAM_IMU_aug = P_aug[0:15,15:P_aug.shape[1]]
        
        Update = EKF_update[i]
        H_o = []

        r_o = []

        R_o = np.empty((0,0))

        for j in range(len(Update)):
            
            cam_states = [element['camState'] for element in groundTruthStates[image_to_index[Update[j][1][0][0]] : image_to_index[Update[j][1][-1][0]] +1]  ]
            
            observations =  np.array([item[2] for item in Update[j][1]]).T

            observations[0,:] = (observations[0,:]-cu)/fu
            observations[1,:] = (observations[1,:]-cv)/fv


            p_f_G , Jnew, Rcond = calc_gn_pos_est(cam_states, observations, u_var_prime, v_var_prime)
            
            
            
           


            r_j = calc_residual(p_f_G, cam_states, observations)
            
            
            
            repeats = np.array([u_var_prime, v_var_prime])

            num_repeats = len(r_j) // 2

            R_j = np.diag(np.tile(repeats, num_repeats))
            
            
            A_j, H_o_j, H_x_ji= Jacobian(p_f_G, Update, cam_states)
            
            
            
            H_o.append(H_o_j)
            
            r_o_j = A_j.T @ r_j
            r_o.append(r_o_j)

            R_o_j = A_j.T @ R_j @ A_j
            
            new_rows = R_o.shape[0] + R_o_j.shape[0]
            new_cols = R_o.shape[1] + R_o_j.shape[1]


             
            new_R_o = np.zeros((new_rows, new_cols))
            new_R_o[:R_o.shape[0], :R_o.shape[1]] = R_o
            new_R_o[R_o.shape[0]:R_o.shape[0] + R_o_j.shape[0], R_o.shape[1]:R_o.shape[1] + R_o_j.shape[1]] = R_o_j
             
            R_o = new_R_o

        H_o = np.vstack(H_o)   
        r_o = np.vstack(r_o)    
        T_H1, Q_1 = T_H(H_o)
        r_n = Q_1.T@ r_o
        R_n = Q_1.T @ R_o @ Q_1

        K = (P_aug@T_H1.T)@ np.linalg.inv( T_H1@P_aug@T_H1.T + R_n )


        deltaX = K @ r_n

        deltatheta_IG = deltaX[0:3]
        deltab_g = deltaX[3:6]
        deltav_I_G = deltaX[6:9]
        deltab_a = deltaX[9:12]
        deltap_I_G = deltaX[12:15]

        deltaq = 0.5 * deltatheta_IG
            
        checkNorm = deltaq.T@ deltaq;
            
        if (checkNorm > 1):
            updateQuat = np.append(deltaq, 1)        
            updateQuat = updateQuat / np.sqrt(1 + checkNorm)
            
        else:
            updateQuat = np.append(deltaq, np.sqrt(1 - checkNorm))
            
                
            
        deltaq_IG = updateQuat / np.linalg.norm(updateQuat)    
            
            
        q_IG_up = quatLeftComp(deltaq_IG.reshape((4,1)))@q_IG
        b_g_up = b_g + deltab_g.reshape((3,))
        b_a_up = b_a + deltab_a.reshape((3,))
        p_I_G_up = p_I_G.reshape((3,1)) + deltap_I_G
        
        window = i +1
        
        q_CG_up = []

        p_C_G_up = []
            

        for k in range(i+1):
            
            qStart = 15 + 6 * k  
            pStart = qStart + 3
            
            deltatheta_CG = deltaX[qStart:qStart + 3]
            deltap_C_G = deltaX[pStart:pStart + 3]
            
            deltaq = 0.5 * deltatheta_CG
            
            checkNorm = deltaq.T@ deltaq;
                
            if (checkNorm > 1):
                updateQuat = np.append(deltaq, 1)        
                updateQuat = updateQuat / np.sqrt(1 + checkNorm)
                
            else:
                updateQuat = np.append(deltaq, np.sqrt(1 - checkNorm))
                
                    
                
            deltaq_CG = updateQuat / np.linalg.norm(updateQuat)    
                
            q_CG_up1 = quatLeftComp(deltaq_CG.reshape((4,1))) @ groundTruthStates[k]['camState']['q_CG']
            
            q_CG_up.append(q_CG_up1)
            
            p_C_G_up1 = groundTruthStates[k]['camState']['p_C_G'].reshape((3,1)) + deltap_C_G
            
            p_C_G_up.append(p_C_G_up1)     
            
        q1.append(q_CG_up1)
        p1.append(p_C_G_up1)
        
        tempMat = (np.eye(15 + 6*window) - K@T_H1)


        P_corrected = tempMat @ P_aug @ tempMat.T + K @ R_n @ K.T

        P_IMU_cor = P_corrected[0:15,0:15]
        P_CAM_cor = P_corrected[15:P_corrected.shape[0],15:P_corrected.shape[1]]
        P_CAM_IMU_cor = P_corrected[0:15, 15:P_corrected.shape[1]]
        
        P_IMU_aug = P_IMU_cor
        P_CAM_IMU_aug = P_CAM_IMU_cor
        P_CAM_aug = P_CAM_cor      
        
        
        bias_a.append(b_a_up)
        bias_g.append(b_g_up)
        q = []
        p = []
        
        for l in range(window):
                         
            q_IG = -quat_mult(q_IC, q_CG_up[l])
                
            C_IG = quatToRotMat(q_IG)
                
            p_C_G = p_C_G_up[l]
            p_I_G = p_C_G + ((quatToRotMat(q_CG_up[l])).T@p_I_C).reshape((3,1))
                
            
                
            
            q.append(q_IG)
            
            p.append(p_I_G)

#%%



kNum = len(EKF_update)
p_C_G_est = np.zeros((3,kNum))
p_I_G_imu = np.zeros((3,kNum))
p_I_G_est = np.zeros((3,kNum))

p_C_G_GT = np.zeros((3,kNum))

q = [entry['camState']['q_CG'] for entry in groundTruthStates]

theta = []
for k in range(kNum):
    p_C_G_GT[:,k] = groundTruthStates[k]['camState']['p_C_G']
    p_C_G_est[:,k] = p_C_G_up[k].reshape((3,))
    q_CG_est  = q_CG_up[k]  
    
    q_CG_est[:3] = -q_CG_est[:3]

    qe = quat_mult(q[k], q_CG_est)
    
    theta1 = np.arctan2(np.sqrt(qe[0]**2 + qe[1]**2 + qe[2]**2),qe[3])
    theta.append(theta1)   
    p_I_G_imu[:,k] = groundTruthStates[k]['imuState']['p_I_G']
    p_I_G_est[:,k] = (p[k]).reshape((3,))
    
    
    
                        
                    



msckf_trans_err = p_C_G_est - p_C_G_GT
msckf_trans_err = msckf_trans_err[:, ~np.isnan(msckf_trans_err).any(axis=0)]

armse_trans_msckf_x = np.sqrt(np.mean(( p_C_G_est[0,:] - p_C_G_GT[0,:])**2))
armse_trans_msckf_y = np.sqrt(np.mean(( p_C_G_est[1,:] - p_C_G_GT[1,:])**2))
armse_trans_msckf_z = np.sqrt(np.mean(( p_C_G_est[2,:] - p_C_G_GT[2,:])**2))







differences = [
    p_C_G_est[0, :] - p_C_G_GT[0, :],
    p_C_G_est[1, :] - p_C_G_GT[1, :],
    p_C_G_est[2, :] - p_C_G_GT[2, :],
  
]



fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2 rows, 2 columns layout

for i in range(2):
    axs[0, i].plot(differences[i], marker='o', linestyle='-', color='blue')
    axs[0, i].set_title(f'Translational error δ{"xyz"[i]}')
    axs[0, i].set_xlabel('Poses')
    axs[0, i].set_ylabel(f'δ{"xyz"[i]} (m)')
    axs[0, i].grid(True)
    axs[0, i].legend()

   
axs[1, 0].plot(differences[2], marker='o', linestyle='-', color='blue')
axs[1, 0].set_title(f'Translational error δz')
axs[1, 0].set_xlabel('Poses')
axs[1, 0].set_ylabel('δz (m)')
axs[1, 0].grid(True)
axs[1, 0].legend()


axs[1, 1].plot(theta, marker='o', linestyle='-', color='green')
axs[1, 1].set_title('Rotation error')
axs[1, 1].set_xlabel('Poses')
axs[1, 1].set_ylabel('Theta (rad)')
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(15, 15)) 
fig.clf()
ax = fig.add_subplot(111, projection='3d')
ax.grid(True)

ax.plot3D(p_I_G_est[0, :], p_I_G_est[1, :], p_I_G_est[2, :], linestyle='--', color='blue', label='MSCKF')
ax.plot3D(p_I_G_imu[0, :], p_I_G_imu[1, :], p_I_G_imu[2, :], linestyle=':', color='red', label='Ground Truth')# Plot IMU integration

ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_zlabel('z(m)')
ax.legend()
plt.show()

bias_a = np.column_stack(bias_a)
bias_g = np.column_stack(bias_g)


fig, axes = plt.subplots(3, 2, figsize=(15, 15))  

titles = ['Accelerometer Bias X', 'Gyroscope Bias X',
          'Accelerometer Bias Y', 'Gyroscope Bias Y',
          'Accelerometer Bias Z', 'Gyroscope Bias Z']

index = 0

for i in range(3):
    ax = axes[i, 0]
    ax.plot(bias_a[i, :928], label=f'Accel Bias {titles[index][-1]}')
    ax.set_title(titles[index])
    ax.set_ylabel('Bias')
    ax.grid(True)
    index += 2  

index = 1

for i in range(3):
    ax = axes[i, 1]
    ax.plot(bias_g[i, :928], label=f'Accel Bias {titles[index][-1]}')
    ax.set_title(titles[index])
    ax.set_ylabel('Bias')
    ax.grid(True)
    index += 2  

for ax in axes[-1, :]:
    ax.set_xlabel('Poses')

fig.tight_layout()

plt.show()
