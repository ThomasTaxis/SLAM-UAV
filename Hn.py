import numpy as np
from scipy.linalg import null_space
from crossVec import crossMat
from qToR import quatToRotMat


def Jacobian(p_f_G, state, cam_states):
    
    
    N = int((int(state[0][1][-1][0].replace('.png', ''))-5)/5 +1) +1
    
    M = len(cam_states)
    
    
    H_fj = np.zeros((2*M,3))
    H_xj = np.zeros((2*M,15+6*N))
    
    
    
    for i in range(len(cam_states)):
        C_CG = quatToRotMat(cam_states[i]['q_CG'])
        p_f_C = C_CG@(p_f_G - cam_states[i]['p_C_G'])
        
        p_f_C = p_f_C.reshape((3,1))
        
        X = p_f_C[0,0]
        Y = p_f_C[1,0]
        Z = p_f_C[2,0]
        
        if (Z == 0):
            
            J_i = np.zeros((2,3))
            
        else:
            J_i = (1/Z)*np.array([[1, 0, -X/Z], [0, 1, -Y/Z]])           
        
        
        
        row_start = 2 * i
        row_end = row_start + 2
        col_start_1 = 15 + 6 * i  # Adjusted to correct start index for the blocks
        col_end_1 = col_start_1 + 3
        col_start_2 = col_end_1
        col_end_2 = col_start_2 + 3

        H_fj[row_start:row_end, :] = J_i @ C_CG
        H_xj[row_start:row_end, col_start_1:col_end_1] = J_i @ crossMat(p_f_C)
        H_xj[row_start:row_end, col_start_2:col_end_2] = -J_i @ C_CG
        
    A_j = null_space(H_fj.T)
    H_o_j = A_j.T@H_xj
    
    
    return A_j, H_o_j,H_xj
    
    
    
    
    
    