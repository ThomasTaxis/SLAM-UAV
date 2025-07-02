import quatLeftComp

def quatMult(quat1, quat2):
    
    if quat1.shape != (4, 1) or quat2.shape != (4, 1):
        raise ValueError("Input quaternions must be 4x1")

    quatProd = quatLeftComp(quat1) @ quat2  # Using matrix multiplication to compute the product
    return quatProd