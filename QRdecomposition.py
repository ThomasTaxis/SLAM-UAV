import numpy as np

def T_H(H_o):
    
    Q,R = np.linalg.qr(H_o, mode='complete')
    
    isZeroRow = np.all(R == 0, axis=1)
    
    # Extract relevant matrices
    T_H = R[~isZeroRow, :]
    Q_1 = Q[:, ~isZeroRow]
    
    return T_H, Q_1