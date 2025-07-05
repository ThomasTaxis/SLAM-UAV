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

DATA_ROOT = 'data/dataset/2011_09_29/2011_09_29_drive_0071_sync'
image_data = DATA_ROOT


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
 
poses2 = {k: v for k, v in filtered_dict.items() if len(v) > 4 }



import json
with open('visualCorrections.json', 'w') as json_file:
    json.dump(filtered_dict, json_file)
    
