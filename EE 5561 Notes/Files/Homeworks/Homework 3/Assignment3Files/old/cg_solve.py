import numpy as np

def cg_solve(b, A, no_iter=3):
    
# ----cg_solve------------------------------------------
#     x_recon = A^-1.b
#     no_iter: numver of iterations
#     x_recon = cg_solve(b,A,no_iter)

    x0 = np.zeros_like(b)
    x = np.copy(x0)
    r = b - A(x)
    p = np.copy(r)
    rsold = np.sum(r * np.conj(r))
    
    for ind in range(no_iter):
        Ap = A(p)
        alpha = rsold/np.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.sum(r * np.conj(r))
        p = r + rsnew / rsold * p
        rsold = np.copy(rsnew)
    return x