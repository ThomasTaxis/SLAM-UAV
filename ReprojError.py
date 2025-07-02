import numpy as np
from qToR import quatToRotMat

def calc_residual(p_f_G, camStates, observations):
    
    r_j =  np.zeros((2 * len(camStates), 1))
    for i, camState in enumerate(camStates):
        C_CG = quatToRotMat(camState['q_CG'])
        p_f_C = C_CG @ (p_f_G - camState['p_C_G'])
        
        if (p_f_C[2]==0):
            zhat_i_j = p_f_C[:2]
        else:
            zhat_i_j = p_f_C[:2] / p_f_C[2]         

       

        iStart = 2 * i
        iEnd = 2 * (i + 1)
        r_j[iStart:iEnd, 0] = observations[:, i] - zhat_i_j
    
    return r_j
