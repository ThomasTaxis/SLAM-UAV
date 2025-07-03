import numpy as np

from qToR import quatToRotMat



def triangulate(obs1, obs2, C_12, t_21_1):
    # Triangulates 3D points from two sets of feature vectors and a frame-to-frame transformation
    v_1 = np.hstack((obs1, [1])).T
    v_2 = np.hstack((obs2, [1])).T
    v_1 /= np.linalg.norm(v_1)
    v_2 /= np.linalg.norm(v_2)

    A = np.concatenate((v_1.reshape((3,1)), (-C_12@v_2).reshape((3,1))),axis=1)
    b = t_21_1.reshape((3,1))

    scalar_consts = np.linalg.lstsq(A, b, rcond=None)[0]
    return scalar_consts[0] * v_1

def calc_gn_pos_est(cam_states, observations, u_var_prime, v_var_prime):
    second_view_idx = len(cam_states) - 1  # Indexing starts from 0 in Python

    C_12 = quatToRotMat(cam_states[0]['q_CG']) @ quatToRotMat(cam_states[second_view_idx]['q_CG']).T
    t_21_1 = quatToRotMat(cam_states[0]['q_CG']) @ (cam_states[second_view_idx]['p_C_G'] - cam_states[0]['p_C_G'])

    p_f1_1_bar = triangulate(observations[:, 0], observations[:, second_view_idx], C_12, t_21_1)

    x_bar, y_bar, z_bar = p_f1_1_bar[:3]
    alpha_bar = x_bar/z_bar
    beta_bar = y_bar/z_bar
    rho_bar = 1/z_bar

    x_est = np.array([alpha_bar, beta_bar, rho_bar])

    Cnum = len(cam_states)
    max_iter = 10
    Jprev = np.inf

    for opt_i in range(max_iter):
        E = np.zeros((2 * Cnum, 3))
        W = np.zeros((2 * Cnum, 2 * Cnum))
        error_vec = np.zeros(2 * Cnum)

        for i_state in range(Cnum):
            W[2 * i_state: 2 * i_state + 2, 2 * i_state: 2 * i_state + 2] = np.diag([u_var_prime, v_var_prime])

            C_i1 = quatToRotMat(cam_states[i_state]['q_CG']) @ quatToRotMat(cam_states[0]['q_CG']).T
            t_1i_i = quatToRotMat(cam_states[i_state]['q_CG']) @ (cam_states[0]['p_C_G'] - cam_states[i_state]['p_C_G'])

            z_hat = observations[:, i_state]
            h = np.dot(C_i1, [alpha_bar, beta_bar, 1]) + rho_bar * t_1i_i

            error_vec[2 * i_state: 2 * i_state + 2] = z_hat - h[:2] / h[2]
            
            dEdalpha = np.array([[-C_i1[0, 0] / h[2] + h[0] / h[2]**2 * C_i1[2, 0]],
                                    [-C_i1[1, 0] / h[2] + h[1] / h[2]**2 * C_i1[2, 0]]])

            dEdbeta = np.array([[-C_i1[0, 1] / h[2] + h[0] / h[2]**2 * C_i1[2, 1]],
                                   [-C_i1[1, 1] / h[2] + h[1] / h[2]**2 * C_i1[2, 1]]])

            dEdrho = np.array([[-t_1i_i[0] / h[2] + h[0] / h[2]**2 * t_1i_i[2]],
                                  [-t_1i_i[1] / h[2] + h[1] / h[2]**2 * t_1i_i[2]]])

            E_block = np.hstack((dEdalpha, dEdbeta, dEdrho))
            E[2 * i_state: 2 * i_state + 2, :] = E_block

            

        Jnew = 0.5 * np.dot(error_vec.T, np.linalg.solve(W, error_vec))
        EWE = E.T @ np.linalg.solve(W, E)
        RCOND = np.linalg.cond(EWE,-2)
        dx_star = np.linalg.solve(EWE, -E.T @ np.linalg.solve(W, error_vec))

        x_est += dx_star

        Jderiv = abs((Jnew - Jprev) / Jnew)
        Jprev = Jnew

        if Jderiv < 0.01:
            break
        else:
            alpha_bar, beta_bar, rho_bar = x_est

    p_f_G = (1 / x_est[2]) * quatToRotMat(cam_states[0]['q_CG']).T @ np.hstack((x_est[:2], [1])) + cam_states[0]['p_C_G']

    return p_f_G, Jnew, RCOND